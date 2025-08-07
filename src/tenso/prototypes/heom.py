# coding: utf-8

from itertools import chain
from typing import Callable, Generator, Literal

from tenso.basis.dvr import SineDVR, SincDVR
from tenso.bath.correlation import Correlation
from tenso.heom.eom import FrameFactory, Hierachy
from tenso.heom.meom import FrameFactory as MBFrameFactory, Hierachy as MBHierachy
from tenso.heom.multieom import FrameFactory as MSMBFrameFactory, Hierachy as MSMBHierachy
from tenso.libs.backend import OptArray, opt_array, opt_linalg, opt_to_numpy
from tenso.libs.logging import Logger
from tenso.libs.quantity import Quantity as __
from tenso.operator.sparse import SparseSPO, SparsePropagator
from tenso.prototypes.default_parameters import default_extension, get_default_kwargs, quantity, value

import numpy as np

from tenso.state.pureframe import End
from tenso.state.puremodel import Model

# Type hinting
VecList = list[complex]
MatList = list[list[complex]]

inverse_temperature_unit = '/K'
time_unit = 'fs'
energy_unit = '/cm'

parameters = get_default_kwargs(['tn', 'heom', 'propagation'])


def spin_boson(
    fname: str,
    # System
    init_rdo: MatList,
    sys_ham: MatList,
    sys_op: MatList,
    # Bath
    bath_correlation: Correlation,
    # Time-dependent field
    td_f: Callable[[float], float] | None = None,
    td_op: MatList | None = None,
    # other settings
    **kwargs,
) -> Generator[float, None, None]:
    """Spin-Boson model using HEOM with tensor network.
    Assuming one bath correlation function.

    Parameters:
    -----------
    fname: str
        The output file name.
    init_rdo: MatList
        The initial reduced density operator.
    h: MatList
        The system Hamiltonian.
    op: MatList
        The system operator in the system-bath interaction hamiltonian.
    bath_correlation: Correlation
        The bath correlation function for HEOM.
    td_f: Callable[[float], float] | None
        The time-dependent field.
    td_op: MatList | None
        The operator associated with the time-dependent field.
    kwargs: dict
        Other settings. See `default_parameters.py` for details.

    Yields:
    -------
    float
        The current time in unit in `default_parameters.py`.
    """
    print(kwargs, flush=True)
    for k, v in kwargs.items():
        if k in parameters:
            parameters[k] = v
        else:
            print(f'Warning: {k} is not a valid parameter; ignored',
                  flush=True)
    print(parameters, flush=True)
    ue = quantity(1.0, 'energy')

    # HEOM frame:
    frame_method = parameters['frame_method']
    htd = FrameFactory(bath_correlation.k_max)
    if frame_method.lower() == 'train':
        frame, root = htd.train()
    elif frame_method.lower().startswith('tree'):
        n_ary = int(frame_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif frame_method.lower() == 'naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No frame_method {frame_method}.')

    # HEOM basis:
    dim = parameters['dim']
    if isinstance(dim, int):
        bath_dims = [dim] * bath_correlation.k_max
    elif isinstance(dim, list):
        assert len(dim) == bath_correlation.k_max
        assert all(isinstance(d, int) for d in dim)
        bath_dims = list(dim)
    else:
        raise NotImplementedError(f'Not dim type {type(dim)}.')

    # Whether to use DVR as the basis for the hierarchy:
    bases = dict()
    if parameters['use_dvr']:
        dvr_types = {
            k: parameters['dvr_type']
            for k in range(bath_correlation.k_max)
        }
        dvr_lengths = {
            k: parameters['dvr_length']
            for k in range(bath_correlation.k_max)
        }
        for _k in dvr_types.keys():
            _type = dvr_types[_k]
            if _type.lower() == 'sinc':
                dvr_cls = SincDVR
            elif _type.lower() == 'sine':
                dvr_cls = SineDVR
            else:
                raise NotImplementedError(f"No basis named as {_type}.")
            _l = dvr_lengths[_k]
            bases[htd.bath_ends[_k]] = dvr_cls(-_l / 2.0, _l / 2.0,
                                               bath_dims[_k])

    init_rdo = np.array(init_rdo)
    sys_dim = init_rdo.shape[0]
    assert init_rdo.shape == (sys_dim, sys_dim)
    hierachy = Hierachy(frame,
                        root,
                        htd.sys_ket_end,
                        htd.sys_bra_end,
                        htd.bath_ends,
                        sys_dim,
                        bath_dims,
                        bases=bases)

    # HEOM state:
    rank = parameters['rank']
    if parameters['load_checkpoint_from_file']:
        state = Model.load(fname + default_extension['checkpoint'])
        renorm_coeff = state[root].norm()
        state.update({root: state[root] / renorm_coeff})
    else:
        state = hierachy.initialize_state(init_rdo, rank)
        renorm_coeff = 1.0
    # HEOM operator:
    heom_metric = parameters['metric']
    if isinstance(heom_metric, str):
        assert heom_metric in ('abs', 're')
        metric = heom_metric
    elif isinstance(heom_metric, float):
        metric = complex(heom_metric)
    elif isinstance(heom_metric, tuple):
        assert len(heom_metric) == 2
        metric = complex(*heom_metric)
    else:
        raise NotImplementedError(f'No heom_factor type {type(heom_metric)}.')
    lvn_list = hierachy.lvn_list(sys_ham * ue)
    heom_list = hierachy.heom_list(sys_op, bath_correlation, metric)
    lindblad_list = hierachy.lindblad_list(sys_op,
                                           bath_correlation.lindblad_rate)

    if td_f is not None and td_op is not None:
        _op = opt_array(td_op)
        zeros = opt_array(np.zeros_like(td_op))
        i_end = hierachy.sys_ket_end
        j_end = hierachy.sys_bra_end

        def f_list(time: float) -> list[dict[End, OptArray]]:
            _f = td_f(time)
            if abs(_f) > 1e-14:
                ans = [
                    {
                        i_end: -1.0j * _f * _op
                    },
                    {
                        j_end: 1.0j * _f.conjugate() * _op.T.conj()
                    },
                ]
            else:
                ans = [
                    {
                        i_end: zeros
                    },
                    {
                        j_end: zeros
                    },
                ]
            return ans
    else:
        f_list = None

    # Propagator:
    start_time = quantity(parameters['start_time'], 'time')

    sp_kwargs = {
        k: v
        for k in SparsePropagator.keyword_settings
        if (v := parameters.get(k)) is not None
    }
    SparsePropagator.update_settings(**sp_kwargs)
    propagator = SparsePropagator(
        SparseSPO(lvn_list + heom_list + lindblad_list,
                  f_list=f_list,
                  initial_time=start_time), state, frame, root)
    propagation_method = parameters['stepwise_method']
    end = quantity(parameters['end_time'], 'time')
    dt = quantity(parameters['step_time'], 'time')
    ps_method = parameters['ps_method']
    if propagation_method == 'simple':
        prop_it = propagator.propagate(end, dt, ps_method)
    elif propagation_method == 'mix':
        if (dt1 := parameters['auxiliary_step_time']) is not None:
            dt1 = quantity(dt1, 'time')
        prop_it = propagator.mixed_propagate(
            end,
            dt,
            ending_ps_method=ps_method,
            starting_dt=dt1,
            starting_ps_method=parameters['auxiliary_ps_method'],
            max_starting_rank=parameters['max_auxiliary_rank'],
            max_starting_steps=parameters['max_auxiliary_steps'],
        )
    else:
        raise NotImplementedError(
            f'No propagation method {propagation_method}.')

    if parameters['visualize_frame'] == True:
        from tenso.libs.drawing import visualize_frame
        visualize_frame(frame, fname=fname)
        print(f"Frame graph is saved as {fname}.pdf")
    # Output logger:
    link_it = frame.node_link_visitor(root)
    tracking_dims = []
    tracking_info = []
    for p, i, q, j in link_it:
        if (q, j) not in tracking_dims:
            tracking_dims.append((p, i))
            tracking_info.append(f"{p}-{q}")
    output_logger = Logger(filename=fname + default_extension['output'],
                           level='info')
    output_logger.info('# time rdo00 rdo01 rdo10 rdo11')
    debug_logger = Logger(filename=fname + default_extension['debug'],
                          level='info')
    debug_logger.info('# ' + propagator.info())
    debug_logger.info(f'# frame = {frame}')
    debug_logger.info('# time ode_steps max_rank tr norm')
    debug_logger.info('# # ' + " ".join(tracking_info))

    renormalize = parameters['renormalize']
    for _t, _s in prop_it:
        time = value(_t, 'time')
        rdo = opt_to_numpy(hierachy.get_rdo(state))
        rdo *= renorm_coeff
        root_array = state[root]
        ranks = [state.dimension(p, i) for p, i in tracking_dims]
        flat = rdo.reshape(-1)
        trace = (np.trace(rdo)).real
        norm = opt_linalg.norm(root_array.reshape(-1)).item()

        output_logger.info(f'{time} ' + " ".join(f'{p_i:.8f}' for p_i in flat))
        debug_logger.info(
            f'{time} {propagator.ode_step_counter} {max(ranks) if ranks else 0} {trace} {norm}'
        )
        if ranks:
            debug_logger.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))

        if renormalize:
            state.update({root: root_array / norm})
            renorm_coeff *= norm

        yield time

    if parameters['save_checkpoint_to_file']:
        if parameters['renormalize']:
            state.update({root: root_array * renorm_coeff})
        state.save(fname + default_extension['checkpoint'])
    return


def system_multibath(
    fname: str,
    # System
    init_rdo: MatList,
    sys_ham: MatList,
    sys_ops: list[MatList],
    # Bath
    bath_correlations: list[Correlation],
    # Time-dependent field
    td_f: Callable[[float], float] | None = None,
    td_op: MatList | None = None,
    # other settings
    **kwargs,
) -> Generator[float, None, None]:
    """Spin-Boson model using HEOM with tensor network.
    Assuming one bath correlation function.

    Parameters:
    -----------
    fname: str
        The output file name.
    init_rdo: MatList
        The initial reduced density operator.
    h: MatList
        The system Hamiltonian.
    op: MatList
        The system operator in the system-bath interaction hamiltonian.
    bath_correlation: Correlation
        The bath correlation function for HEOM.
    td_f: Callable[[float], float] | None
        The time-dependent field.
    td_op: MatList | None
        The operator associated with the time-dependent field.
    kwargs: dict
        Other settings. See `default_parameters.py` for details.

    Yields:
    -------
    float
        The current time in unit in `default_parameters.py`.
    """
    for _k, v in kwargs.items():
        if _k in parameters:
            parameters[_k] = v
        else:
            print(f'Warning: {_k} is not a valid parameter; ignored',
                  flush=True)
    print(parameters, flush=True)
    ue = quantity(1.0, 'energy')

    # HEOM frame:
    frame_method = parameters['frame_method']
    htd = MBFrameFactory([c.k_max for c in bath_correlations])
    if frame_method.lower() == 'train':
        frame, root = htd.train()
    elif frame_method.lower().startswith('tree'):
        n_ary = int(frame_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif frame_method.lower() == 'naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No frame_method {frame_method}.')

    # HEOM basis:
    dim = parameters['dim']
    if isinstance(dim, int):
        bath_dims = [[dim] * c.k_max for c in bath_correlations]
    elif isinstance(dim, list):
        assert len(dim) == len(bath_correlations)
        if isinstance(dim[0], int):
            bath_dims = [[d] * c.k_max for d, c in zip(dim, bath_correlations)]
        else:
            assert isinstance(dim[0], list)
            bath_dims = list()
            for ds, c in zip(dim, bath_correlations):
                assert len(ds) == c.k_max
                bath_dims.append(list(ds))
    else:
        raise NotImplementedError(f'Not dim type {type(dim)}.')

    # Whether to use DVR as the basis for the hierarchy:
    bases = dict()
    if parameters['use_dvr']:
        dvr_types = {
            (n, k): parameters['dvr_type']
            for n, c in enumerate(bath_correlations)
            for k in range(c.k_max)
        }
        dvr_lengths = {
            (n, k): parameters['dvr_type']
            for n, c in enumerate(bath_correlations)
            for k in range(c.k_max)
        }
        for (_n, _k) in dvr_types.keys():
            _type = dvr_types[_n, _k]
            if _type.lower() == 'sinc':
                dvr_cls = SincDVR
            elif _type.lower() == 'sine':
                dvr_cls = SineDVR
            else:
                raise NotImplementedError(f"No basis named as {_type}.")
            _l = dvr_lengths[_k]
            bases[htd.bath_ends[_n][_k]] = dvr_cls(-_l / 2.0, _l / 2.0,
                                                   bath_dims[_k])

    init_rdo = np.array(init_rdo)
    sys_dim = init_rdo.shape[0]
    assert init_rdo.shape == (sys_dim, sys_dim)
    hierachy = MBHierachy(frame,
                          root,
                          htd.sys_ket_end,
                          htd.sys_bra_end,
                          htd.bath_ends,
                          sys_dim,
                          bath_dims,
                          bases=bases)

    # HEOM state:
    rank = parameters['rank']
    if parameters['load_checkpoint_from_file']:
        state = Model.load(fname + default_extension['checkpoint'])
        renorm_coeff = state[root].norm()
        state.update({root: state[root] / renorm_coeff})
    else:
        state = hierachy.initialize_state(init_rdo, rank)
        renorm_coeff = 1.0
    # HEOM operator:
    heom_metric = parameters['metric']
    if isinstance(heom_metric, str):
        assert heom_metric in ('abs', 're')
        metric = heom_metric
    elif isinstance(heom_metric, float):
        metric = complex(heom_metric)
    elif isinstance(heom_metric, tuple):
        assert len(heom_metric) == 2
        metric = complex(*heom_metric)
    else:
        raise NotImplementedError(f'No heom_factor type {type(heom_metric)}.')
    lvn_list = hierachy.lvn_list(sys_ham * ue)
    heom_list = []
    lindblad_list = []
    for n, bath_correlation in enumerate(bath_correlations):
        heom_list += hierachy.heom_list(n, sys_ops[n], bath_correlation,
                                        metric)
        lindblad_list += hierachy.lindblad_list(sys_ops[n],
                                                bath_correlation.lindblad_rate)

    if td_f is not None and td_op is not None:
        _op = opt_array(td_op)
        zeros = opt_array(np.zeros_like(td_op))
        i_end = hierachy.sys_ket_end
        j_end = hierachy.sys_bra_end

        def f_list(time: float) -> list[dict[End, OptArray]]:
            _f = td_f(time)
            if abs(_f) > 1e-14:
                ans = [
                    {
                        i_end: -1.0j * _f * _op
                    },
                    {
                        j_end: 1.0j * _f.conjugate() * _op.T.conj()
                    },
                ]
            else:
                ans = [
                    {
                        i_end: zeros
                    },
                    {
                        j_end: zeros
                    },
                ]
            return ans
    else:
        f_list = None

    # Propagator:
    sp_kwargs = {
        k: v
        for k in SparsePropagator.keyword_settings
        if (v := parameters.get(k)) is not None
    }
    SparsePropagator.update_settings(**sp_kwargs)
    start_time = quantity(parameters['start_time'], 'time')
    propagator = SparsePropagator(
        SparseSPO(lvn_list + heom_list + lindblad_list,
                  f_list=f_list,
                  initial_time=start_time), state, frame, root)
    propagation_method = parameters['stepwise_method']
    end = quantity(parameters['end_time'], 'time')
    dt = quantity(parameters['step_time'], 'time')
    ps_method = parameters['ps_method']
    if propagation_method == 'simple':
        prop_it = propagator.propagate(end, dt, ps_method)
    elif propagation_method == 'mix':
        if (dt1 := parameters['auxiliary_step_time']) is not None:
            dt1 = quantity(dt1, 'time')
        prop_it = propagator.mixed_propagate(
            end,
            dt,
            ending_ps_method=ps_method,
            starting_dt=dt1,
            starting_ps_method=parameters['auxiliary_ps_method'],
            max_starting_rank=parameters['max_auxiliary_rank'],
            max_starting_steps=parameters['max_auxiliary_steps'],
        )
    else:
        raise NotImplementedError(
            f'No propagation method {propagation_method}.')

    # Output logger:
    link_it = frame.node_link_visitor(root)
    tracking_dims = []
    tracking_info = []
    for p, i, q, j in link_it:
        if (q, j) not in tracking_dims:
            tracking_dims.append((p, i))
            tracking_info.append(f"{p}-{q}")
    output_logger = Logger(filename=fname + default_extension['output'],
                           level='info')
    output_logger.info('# time rdo00 rdo01 rdo10 rdo11')
    debug_logger = Logger(filename=fname + default_extension['debug'],
                          level='info')
    debug_logger.info('# ' + propagator.info())
    debug_logger.info(f'# frame = {frame}')
    debug_logger.info('# time ode_steps max_rank tr norm')
    debug_logger.info('# # ' + " ".join(tracking_info))

    renormalize = parameters['renormalize']
    for _t, _s in prop_it:
        time = value(_t, 'time')
        rdo = opt_to_numpy(hierachy.get_rdo(state))
        rdo *= renorm_coeff
        root_array = state[root]
        ranks = [state.dimension(p, i) for p, i in tracking_dims]
        flat = rdo.reshape(-1)
        trace = (np.trace(rdo)).real
        norm = opt_linalg.norm(root_array.reshape(-1)).item()

        output_logger.info(f'{time} ' + " ".join(f'{p_i:.8f}' for p_i in flat))
        debug_logger.info(
            f'{time} {propagator.ode_step_counter} {max(ranks) if ranks else 0} {trace} {norm}'
        )
        if ranks:
            debug_logger.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))

        if renormalize:
            state.update({root: root_array / norm})
            renorm_coeff *= norm

        yield time

    if parameters['save_checkpoint_to_file']:
        if parameters['renormalize']:
            state.update({root: root_array * renorm_coeff})
        state.save(fname + default_extension['checkpoint'])
    return


def holstein_model(
    fname: str,
    # System
    init_wfns: list[VecList],
    sys_hs: list[None | MatList],
    sys_couplings: list[dict[int, MatList]],
    sys_ops: list[dict[int, MatList]],
    # Bath
    bath_correlations: list[Correlation],
    # Time-dependent field
    tracking_indices: list[tuple[int, int]],
    td_fields: list[Callable[[float], float]] | None = None,
    td_ops: list[dict[int, MatList]] | None = None,
    # other settings
    **kwargs,
) -> Generator[float, None, None]:
    """Holstein model using HEOM with tensor network.
    Assuming multiple bath correlation functions.
    """
    for k, v in kwargs.items():
        if k in parameters:
            parameters[k] = v
        else:
            print(f'Warning: {k} is not a valid parameter; ignored',
                  flush=True)
    print(parameters, flush=True)
    ue = quantity(1.0, 'energy')

    # System settings:
    sys_dof = len(init_wfns)
    bath_dofs = [c.k_max for c in bath_correlations]

    # HEOM frame:
    frame_method = parameters['frame_method']
    htd = MSMBFrameFactory(sys_dof, bath_dofs)
    if frame_method.lower() == 'train':
        frame, root = htd.train()
    elif frame_method.lower().startswith('tree'):
        n_ary = int(frame_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif frame_method.lower() == 'naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No frame_method {frame_method}.')

    # dimension settings:
    sys_dims = list(len(v) for v in init_wfns)
    dim = parameters['dim']
    if isinstance(dim, int):
        bath_dims = [[dim] * c.k_max for c in bath_correlations]
    elif isinstance(dim, list):
        assert len(dim) == len(bath_correlations)
        if isinstance(dim[0], int):
            bath_dims = [[d] * c.k_max for d, c in zip(dim, bath_correlations)]
        else:
            assert isinstance(dim[0], list)
            bath_dims = list()
            for ds, c in zip(dim, bath_correlations):
                assert len(ds) == c.k_max
                bath_dims.append(list(ds))
    else:
        raise NotImplementedError(f'Not dim type {type(dim)}.')

    if parameters['use_dvr']:
        bath_bases = dict()
        for n, bath_correlation in enumerate(bath_correlations):
            dvr_pair = {
                k: (parameters[f'dvr_type_{n}'], parameters[f'dvr_length_{n}'])
                for k in range(bath_correlation.k_max)
            }

            for k, (_type, _l) in dvr_pair.items():
                if _type.lower() == 'sinc':
                    dvr_cls = SincDVR
                elif _type.lower() == 'sine':
                    dvr_cls = SineDVR
                else:
                    raise NotImplementedError(f"No basis named as {_type}.")
                bath_bases[htd.bath_ends[n][k]] = dvr_cls(
                    -_l / 2.0, _l / 2.0, bath_dims[n][k])
    else:
        bath_bases = None

    # HEOM basis:
    hierachy = MSMBHierachy(frame,
                            root,
                            htd.sys_ket_ends,
                            htd.sys_bra_ends,
                            htd.bath_ends,
                            sys_dims,
                            bath_dims,
                            bases=bath_bases)

    # HEOM state:
    rank = parameters['rank']
    if parameters['load_checkpoint_from_file']:
        state = Model.load(fname + default_extension['checkpoint'])
        renorm_coeff = complex(state[root].norm())
        state.update({root: state[root] / renorm_coeff})
    else:
        renorm_coeff = 1.0
        state = hierachy.initialize_pure_state(init_wfns, rank, sys_hs)

    # HEOM settings:
    heom_metric = parameters['metric']
    if isinstance(heom_metric, str):
        assert heom_metric in ('abs', 're')
        metric = heom_metric
    elif isinstance(heom_metric, float):
        metric = complex(heom_metric)
    elif isinstance(heom_metric, tuple):
        assert len(heom_metric) == 2
        metric = complex(*heom_metric)
    else:
        raise NotImplementedError(f'No heom_factor type {type(heom_metric)}.')
    hs_with_ue = [h * ue for h in sys_hs if h is not None]
    coupling_with_ue = []
    for term in sys_couplings:
        new_term = dict()
        length = len(term)
        for m, (k, v) in enumerate(term.items()):
            if m == length - 1:
                new_term[k] = v * ue
            else:
                new_term[k] = v
        coupling_with_ue.append(new_term)
    lvn_list = hierachy.lvn_list(hs_with_ue, sys_couplings=coupling_with_ue)
    heom_list = hierachy.heom_list(sys_ops, bath_correlations, metric)
    lindblad_list = hierachy.lindblad_list(
        sys_ops, [c.lindblad_rate for c in bath_correlations])

    if td_fields and td_ops:
        td_opi = [{
            _i: opt_array(_op)
            for _i, _op in term.items()
        } for term in td_ops]
        td_opj = [{
            _i: _op.T.conj()
            for _i, _op in term.items()
        } for term in td_opi]
        i_ends = hierachy.sys_ket_ends
        j_ends = hierachy.sys_bra_ends

        def f_list(time: float):
            fields = [_f(time) for _f in td_fields]
            ans = []
            for i_end, j_end, _opi, _opj, f in zip(i_ends, j_ends, td_opi,
                                                   td_opj, fields):
                ans += [{
                    i_end: -1.0j * f * _opi
                }, {
                    j_end: 1.0j * f.conjugate() * _opj
                }]
            return ans
    else:
        f_list = None

    # Propagator:
    sp_kwargs = {
        k: v
        for k in SparsePropagator.keyword_settings
        if (v := parameters.get(k)) is not None
    }
    SparsePropagator.update_settings(**sp_kwargs)
    propagator = SparsePropagator(
        SparseSPO(lvn_list + lindblad_list + heom_list,
                  f_list=f_list,
                  initial_time=0.0), state, frame, root)
    propagation_method = parameters['stepwise_method']
    end = quantity(parameters['end_time'], 'time')
    dt = quantity(parameters['step_time'], 'time')
    ps_method = parameters['ps_method']
    if propagation_method == 'simple':
        prop_it = propagator.propagate(end, dt, ps_method)
    elif propagation_method == 'mix':
        if (dt1 := parameters['auxiliary_step_time']) is not None:
            dt1 = quantity(dt1, 'time')
        prop_it = propagator.mixed_propagate(
            end,
            dt,
            ending_ps_method=ps_method,
            starting_dt=dt1,
            starting_ps_method=parameters['auxiliary_ps_method'],
            max_starting_rank=parameters['max_auxiliary_rank'],
            max_starting_steps=parameters['max_auxiliary_steps'],
        )
    else:
        raise NotImplementedError(
            f'No propagation method {propagation_method}.')

    # Output logger:
    link_it = frame.node_link_visitor(root)
    tracking_dims = []
    tracking_info = []
    for p, i, q, j in link_it:
        if (q, j) not in tracking_dims:
            tracking_dims.append((p, i))
            tracking_info.append(f"{p}-{q}")
    output_logger = Logger(filename=fname + default_extension['output'],
                           level='info')

    print(tracking_indices)
    string = ' '.join('(' + '.'.join(str(_i) for _i in _is) + ' , ' +
                      '.'.join(str(_j) for _j in _js) + ')'
                      for _is, _js in tracking_indices)
    output_logger.info('# time ' + string)
    debug_logger = Logger(filename=fname + default_extension['debug'],
                          level='info')
    debug_logger.info('# ' + propagator.info())
    debug_logger.info(f'# frame = {frame}')
    debug_logger.info('# time ode_steps max_rank tr norm')
    debug_logger.info('# # ' + " ".join(tracking_info))

    renormalize = parameters['renormalize']

    len_track = len(tracking_indices)
    rdo_out = np.empty((len_track, ), dtype=np.complex128)
    for _t, _s in prop_it:
        time = value(_t, 'time')
        ranks = [state.dimension(p, i) for p, i in tracking_dims]
        for _n, (_is, _js) in enumerate(tracking_indices):
            rdo_out[_n] = hierachy.get_rdo_element(_s, _is, _js)
        if renormalize:
            rdo_out *= renorm_coeff
        root_array = state[root]
        norm = opt_linalg.norm(root_array.reshape(-1)).item()

        output_logger.info(f"{time} " + " ".join(f"{_r}" for _r in rdo_out))
        debug_logger.info(
            f'{time} {propagator.ode_step_counter} {max(ranks) if ranks else 0}'
        )
        if ranks:
            debug_logger.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))

        yield time

        if renormalize:
            state.update({root: root_array / norm})
            renorm_coeff *= norm

    if parameters['save_checkpoint_to_file']:
        if parameters['renormalize']:
            state.update({root: root_array * renorm_coeff})
        state.save(fname + default_extension['checkpoint'])
    return


def holstein_model_old(
    out: str,
    unit_energy: float,
    # System
    init_wfns: list[VecList],
    sys_hs: list[None | MatList],
    sys_couplings: list[dict[int, MatList]],
    # Bath
    sys_ops: list[dict[int, MatList]],
    bath_correlations: list[Correlation],
    dim: int | list[int] | list[list[int]],
    # HEOM type
    heom_factor: None | str | float | tuple[float, float],
    # Tensor network settings
    htd_method: str,  # Hierachical Tucker Decompositon
    rank: int,  # initial rank
    # Propagator
    tracking_indices: list[tuple[list[int], list[int]]],
    propagation_method: str,
    ps_method: str,
    ode_method: str,
    dt: float,
    end: float,
    # Error control
    ode_rtol: float,
    ode_atol: float,
    ps2_atol: float,
    ps2_ratio: float,
    reg_atol: float,
    # TD
    td_fields: list[Callable[[float], float]],
    td_ops: list[dict[int, MatList]],
    # Debug
    initial_state_from_file: None | str = None,
    **kwargs,
) -> Generator[float, None, None]:

    energy_scale = __(unit_energy, energy_unit).au

    # System settings:
    sys_dof = len(init_wfns)
    sys_dims = list(len(v) for v in init_wfns)
    bath_dofs = [c.k_max for c in bath_correlations]
    if isinstance(dim, int):
        bath_dims = [[dim] * c.k_max for c in bath_correlations]
    elif isinstance(dim, list):
        assert len(dim) == len(bath_correlations)
        if isinstance(dim[0], int):
            bath_dims = [[d] * c.k_max for d, c in zip(dim, bath_correlations)]
        else:
            assert isinstance(dim[0], list)
            bath_dims = list()
            for ds, c in zip(dim, bath_correlations):
                assert len(ds) == c.k_max
                bath_dims.append(list(ds))
    else:
        raise NotImplementedError(f'Not dim type {type(dim)}.')

    # HEOM frame:
    htd = MSMBFrameFactory(sys_dof, bath_dofs)
    if htd_method == 'Train':
        end_order = list(
            chain(htd.sys_ket_ends, *htd.bath_ends,
                  reversed(htd.sys_bra_ends)))
        frame, root = htd.train(end_order)
    elif htd_method.startswith('Tree'):
        n_ary = int(htd_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif htd_method == 'Naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No htd_method {htd_method}.')

    # HEOM basis:
    hierachy = MSMBHierachy(frame,
                            root,
                            htd.sys_ket_ends,
                            htd.sys_bra_ends,
                            htd.bath_ends,
                            sys_dims,
                            bath_dims,
                            bases=None)

    # HEOM state:
    if initial_state_from_file is not None:
        state = Model.load(initial_state_from_file)
    else:
        state = hierachy.initialize_pure_state(init_wfns, rank, sys_hs)

    # HEOM settings:
    if isinstance(heom_factor, str):
        assert heom_factor in ('abs', 're')
        metric = heom_factor
    elif isinstance(heom_factor, float):
        metric = complex(heom_factor)
    elif isinstance(heom_factor, tuple):
        assert len(heom_factor) == 2
        metric = complex(*heom_factor)
    else:
        raise NotImplementedError(f'No heom_factor type {type(heom_factor)}.')
    lvn_list = hierachy.lvn_list(sys_hs, sys_couplings=sys_couplings)
    heom_list = hierachy.heom_list(sys_ops, bath_correlations, metric)
    lindblad_list = hierachy.lindblad_list(
        sys_ops, [c.lindblad_rate for c in bath_correlations])

    if td_fields and td_ops:
        td_opi = [{
            _i: opt_array(_op)
            for _i, _op in term.items()
        } for term in td_ops]
        td_opj = [{
            _i: _op.T.conj()
            for _i, _op in term.items()
        } for term in td_opi]
        i_ends = hierachy.sys_ket_ends
        j_ends = hierachy.sys_bra_ends

        def f_list(time: float):
            fields = [_f(time) for _f in td_fields]
            ans = []
            for i_end, j_end, _opi, _opj, f in zip(i_ends, j_ends, td_opi,
                                                   td_opj, fields):
                ans += [{
                    i_end: -1.0j * f * _opi
                }, {
                    j_end: 1.0j * f.conjugate() * _opj
                }]
            return ans
    else:
        f_list = None

    # Propagator:
    sp_kwargs = {
        k: v
        for k in SparsePropagator.keyword_settings
        if (v := parameters.get(k)) is not None
    }
    SparsePropagator.update_settings(**sp_kwargs)
    propagator = SparsePropagator(
        SparseSPO(lvn_list + lindblad_list + heom_list,
                  f_list=f_list,
                  initial_time=0.0),
        state,
        frame,
        root,
    )
    if propagation_method == 'mix':
        if (dt1 := kwargs.get('starting_dt')) is not None:
            dt1 = __(dt1, time_unit).au * energy_scale
        prop_it = propagator.mixed_propagate(
            __(end, time_unit).au * energy_scale,
            __(dt, time_unit).au * energy_scale,
            ending_ps_method=ps_method,
            starting_dt=dt1,
            starting_ps_method=kwargs.get('starting_method', 'ps2'),
            max_starting_rank=kwargs.get('max_starting_rank'),
            max_starting_steps=kwargs.get('max_starting_steps'),
        )
    elif propagation_method == 'simple':
        prop_it = propagator.propagate(
            __(end, time_unit).au * energy_scale,
            __(dt, time_unit).au * energy_scale,
            ps_method=ps_method,
        )
    elif propagation_method == 'adaptive':
        if (dt1 := kwargs.get('starting_dt')) is not None:
            dt1 = __(dt1, time_unit).au * energy_scale
        prop_it = propagator.adaptive_propagate(
            __(end, time_unit).au * energy_scale,
            __(dt, time_unit).au * energy_scale,
            fixed_ps_method=ps_method,
            fixed_steps=kwargs.get('fixed_steps', 1),
            adaptive_dt=dt1,
            adaptive_ps_steps=kwargs.get('adaptive_steps', 1),
        )
    else:
        raise NotImplementedError(
            f'No propagation method {propagation_method}.')

    state.save(out + '.init.pt')

    # Output logger:
    au2fs = __(1.0).convert_to("fs").value
    link_it = frame.node_link_visitor(root)
    tracking_dims = []
    tracking_info = []
    for p, i, q, j in link_it:
        if (q, j) not in tracking_dims:
            tracking_dims.append((p, i))
            tracking_info.append(f"{p}-{q}")
    logger1 = Logger(filename=out + '.dat.log', level='info')
    string = ' '.join('(' + '.'.join(str(_i) for _i in _is) + ', ' +
                      '.'.join(str(_j) for _j in _js) + ')'
                      for _is, _js in tracking_indices)
    logger1.info('# time ' + string)
    logger2 = Logger(filename=out + '.debug.log', level='info')
    logger2.info("# " + propagator.info())
    logger2.info(f'# Frame structure:')
    logger2.info(f'# {frame}')
    logger2.info('# time ode_steps max_ps_rank')
    logger2.info('# # ' + " ".join(tracking_info))
    len_track = len(tracking_indices)

    rdo_out = np.empty((len_track, ), dtype=np.complex128)
    for _t, _s in prop_it:
        t = _t / energy_scale
        for _n, (_is, _js) in enumerate(tracking_indices):
            rdo_out[_n] = hierachy.get_rdo_element(_s, _is, _js)
        ranks = [state.dimension(p, i) for p, i in tracking_dims]

        logger1.info(f"{t} " + " ".join(f"{_r}" for _r in rdo_out))
        logger2.info(
            f'{t} {propagator.ode_step_counter} {max(ranks) if ranks else 0}')
        if ranks:
            logger2.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))

        yield (t * au2fs)
    state.save(out + '.final.pt')
    return


# def run_dvr(
#     out: str,
#     # System
#     ## Elec
#     init_wfn: Array,
#     elec_bias: float,
#     pes_frequency: float,
#     elec_coupling: float,
#     ## Nuc
#     dvr_space: tuple[float, float],
#     dvr_dim: int,
#     # Drudian bath
#     include_drude: bool,
#     re_d: Optional[float],
#     width_d: float,
#     # LTC bath
#     temperature: float,
#     decomposition_method: str,
#     n_ltc: int,
#     # Tensor Hierachy Tucker Decompositon
#     dim: int,
#     htd_method: str,
#     # HEOM type
#     heom_factor: float,
#     ode_method: str,
#     ps_method: str,
#     reg_method: str,
#     # Error
#     roundoff: float,
#     ode_rtol: float,
#     ode_atol: float,
#     ps2_atol: float,
#     # Propagator
#     dt: tuple[float, float],
#     end: tuple[float, float],
#     callback_steps: tuple[int, float],
# ) -> Generator[tuple[float, OptArray], None, None]:

#     backend.parameters.ode_rtol = ode_rtol
#     backend.parameters.ode_atol = ode_atol
#     backend.parameters.ps2_atol = ps2_atol

#     # System settings:
#     proj_0 = backend.as_array([[1.0, 0.0], [0.0, 0.0]])
#     proj_1 = backend.as_array([[0.0, 0.0], [0.0, 1.0]])
#     sigma_z = backend.as_array([[-0.5, 0.0], [0.0, 0.5]])
#     sigma_x = backend.as_array([[0.0, 1.0], [1.0, 0.0]])

#     # Elec-Nuc
#     def left_morse(depth, frequency, center):
#         alpha = np.sqrt(frequency / 2.0 * depth)

#         def func(x):
#             return depth * (1.0 - np.exp(-alpha * (x - center)))**2

#         return func

#     def right_morse(depth, frequency, center):
#         alpha = np.sqrt(frequency / 2.0 * depth)

#         def func(x):
#             return depth * (1.0 - np.exp(alpha * (x - center)))**2

#         return func

#     def gaussian(height, sigma, center, phase=None):

#         def func(x):
#             ans = height * np.exp(-(x - center)**2 / (2.0 * sigma**2))
#             if phase is not None:
#                 ans *= np.exp(-1.0j * x * phase)
#             return ans

#         return func

#     basis = dvr.SineDVR(dvr_space[0], dvr_space[1], dvr_dim)
#     kinetic = np.tensordot(np.identity(2), basis.t_mat,
#                            axes=0).swapaxes(1,
#                                             2).reshape(2 * dvr_dim, 2 * dvr_dim)
#     e0 = np.tensordot(proj_0,
#                       np.diag(
#                           left_morse(elec_bias, pes_frequency,
#                                      -3)(basis.grid_points)),
#                       axes=0).swapaxes(1, 2).reshape(2 * dvr_dim, 2 * dvr_dim)
#     e1 = np.tensordot(proj_1,
#                       np.diag(
#                           right_morse(elec_bias, pes_frequency,
#                                       3)(basis.grid_points)),
#                       axes=0).swapaxes(1, 2).reshape(2 * dvr_dim, 2 * dvr_dim)
#     v = np.tensordot(sigma_x,
#                      np.diag(
#                          gaussian(elec_coupling, 1.0, 0)(basis.grid_points)),
#                      axes=0).swapaxes(1, 2).reshape(2 * dvr_dim, 2 * dvr_dim)

#     wfn = np.tensordot(init_wfn,
#                        gaussian(1.0, np.sqrt(1 / pes_frequency),
#                                 -10)(basis.grid_points),
#                        axes=0).reshape(-1)
#     wfn /= np.linalg.norm(wfn)
#     init_rdo = np.outer(wfn, wfn)
#     h = kinetic + e0 + e1 + v
#     op = np.tensordot(sigma_z, np.identity(dvr_dim),
#                       axes=0).swapaxes(1, 2).reshape(2 * dvr_dim, 2 * dvr_dim)

#     # Bath settings:
#     distr = BoseEinstein(n=n_ltc, beta=1 / temperature)
#     distr.decomposition_method = decomposition_method
#     sds = []  # type:list[SpectralDensity]
#     if include_drude:
#         drude = Drude(re_d, width_d)
#         sds.append(drude)
#     corr = Correlation(sds, distr)
#     corr.fix(roundoff=roundoff)
#     # print(corr)

#     # HEOM settings:
#     dims = [dim] * corr.k_max
#     if htd_method == 'Naive':
#         s = NaiveHierachy(init_rdo, dims)
#     else:
#         raise NotImplementedError(f'No htd_method {htd_method}.')
#     HeomOp.scaling_factor = heom_factor
#     heom_op = HeomOp(s, h, op, corr, dims)

#     # Propagator settings:
#     steps = int(end / dt)
#     interval = dt / callback_steps
#     propagator = Propagator(heom_op,
#                             s,
#                             interval,
#                             ode_method=ode_method,
#                             ps_method=ps_method,
#                             reg_method=reg_method)

#     logger = Logger(filename=out, level='info')

#     for _n, _t in zip(range(steps), propagator):
#         rdo = s.get_rdo()
#         pop = np.diag(rdo)
#         rdo = rdo.reshape(2, dvr_dim, 2, dvr_dim)
#         # rho_s = np.einsum('ikjk->ij', rdo)
#         # print(rho_s, flush=True)
#         # s.opt_update(s.root, s[s.root] / trace)
#         t = _t
#         logger.info(f'{t} ' + ' '.join(pop))

#     return
