# coding: utf-8

from itertools import chain
from typing import Callable, Generator, Literal

from tenso.basis.dvr import SineDVR, SincDVR, DiscreteVariationalRepresentation as Dvr
from tenso.bath.correlation import Correlation
from tenso.qle.eom import FrameFactory, Hierachy
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

parameters = get_default_kwargs(['tn', 'qle', 'propagation'])


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
    """Spin-Boson model using HEOM in a *QLE* representation with tensor network.
    Assuming one bath.

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
    include_quasi_lindblad: bool
        Whether to include quasi-lindblad terms in the HEOM.
        Calculated from the imaginary system-bath coupling.
        If False, the imaginary part of the system-bath coupling is ignored.
        If True, the quasi-lindblad terms are included in the HEOM, 
        but the positivity of the system's density operator is not guaranteed.
        default is True.
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
    bases = dict() # type: dict[int, Dvr]
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
            bases[_k] = dvr_cls(-_l / 2.0, _l / 2.0,
                                               bath_dims[_k])
    init_rdo = np.array(init_rdo)
    sys_dim = init_rdo.shape[0]
    assert init_rdo.shape == (sys_dim, sys_dim)
    hierachy = Hierachy(frame,
                        root,
                        htd.sys_ket_end,
                        htd.sys_bra_end,
                        htd.bath_ket_ends,
                        htd.bath_bra_ends,
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
    lvn_list = hierachy.lvn_list(np.array(sys_ham) * ue)
    heom_list = hierachy.quasi_lindblad_list(sys_op, bath_correlation, hermite=parameters['hermite'])

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
        SparseSPO(lvn_list + heom_list,
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
