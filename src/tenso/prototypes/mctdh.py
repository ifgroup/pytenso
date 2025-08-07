# coding: utf-8
"""MCTDH prototype for the spin-boson model.
"""
from math import ceil, floor
from typing import Callable, Generator

import numpy as np

#from mugnier.basis.dvr import SincDVR, SineDVR
from tenso.bath.star import StarBosons
from tenso.mctdh.eom import FrameFactory, Hierachy
from tenso.libs.backend import OptArray, opt_array, opt_to_numpy
from tenso.libs.logging import Logger
from tenso.libs.quantity import Quantity as __
from tenso.operator.sparse import ListModelInnerProduct, SparseSandwich, SPOKet, SparsePropagator, SparseSPO
from tenso.prototypes.default_parameters import default_extension, get_default_kwargs, quantity, value
from tenso.state.pureframe import End
from tenso.state.puremodel import Model

# Type hinting
VecList = list[complex]
MatList = list[list[complex]]
parameters = get_default_kwargs(['tn', 'mctdh', 'propagation'])


def spin_boson(
    fname: str,
    # System
    init_wfn: VecList,
    sys_ham: MatList,
    sys_op: MatList,
    # Bath
    bath: StarBosons,
    # Time-dependent fields: TBD
    td_f: Callable[[float], float] | None = None,
    td_op: MatList | None = None,
    # others
    **kwargs,
) -> Generator[float, None, None]:
    """
    Spin-boson model with MCTDH.
    
    Parameters
    ----------
    fname : str
        The filename prefix for the output files.
    init_wfn : VecList
        The initial wavefunction.
    h : MatList
        The system Hamiltonian.
    op : MatList
        The observable.
    boson_bath : StarBosons
        The boson bath.
    td_f : Callable[[float], float], optional
        The time-dependent field, by default None.
    td_op : MatList, optional
        The time-dependent operator, by default None.
    **kwargs
        The parameters for the simulation.

    Yields
    ------
    float
        Current time in unit in `default_parameters.py`.
    """
    for k, v in kwargs.items():
        if k in parameters:
            parameters[k] = v
        else:
            print(f'Warning: {k} is not a valid parameter; ignored',
                  flush=True)
    print(parameters, flush=True)

    # Frame settings:
    frame_method = parameters['frame_method']
    htd = FrameFactory(1, [bath.k_max])
    if frame_method.lower() == 'train':
        frame, root = htd.train()
    elif frame_method.lower().startswith('tree'):
        n_ary = int(frame_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif frame_method.lower() == 'naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No htd_method {frame_method}.')

    # Boson bath settings:
    dim = parameters['dim']
    if isinstance(dim, int):
        bath_dims = [dim] * bath.k_max
    elif isinstance(dim, list):
        assert len(dim) == bath.k_max
        assert all(isinstance(d, int) for d in dim)
        bath_dims = list(dim)
    else:
        raise NotImplementedError(f'Not dim type {type(dim)}.')

    hierachy = Hierachy(frame, root, htd.sys_ends, htd.bath_ends, [2],
                        [bath_dims])

    # Connect the system part and the bath branch
    ue = quantity(1.0, 'energy')
    print('System H:\n', sys_ham, flush=True)
    rank = parameters['rank']
    if parameters['load_checkpoint_from_file']:
        state = Model.load(fname + default_extension['checkpoint'])
        renorm_coeff = state[root].norm()
        state.update({root: state[root] / renorm_coeff})
    else:
        state = hierachy.initialize_pure_state([init_wfn],
                                               rank=rank,
                                               local_hs=[sys_ham * ue])
        renorm_coeff = 1.0
    tdse_list = hierachy.tdse_list(sys_hamiltonians=[sys_ham * ue],
                                   sys_couplings=[])
    heom_list = hierachy.heom_list([{0: sys_op}], [bath])

    if td_f is not None and td_op is not None:
        _op = opt_array(td_op)
        zeros = opt_array(np.zeros_like(td_op))

        def f_list(time: float) -> list[dict[End, OptArray]]:
            _f = td_f(time)
            _i_end = hierachy.sys_ends[0]
            if abs(_f) > 1e-14:
                ans = [
                    {
                        _i_end: -1.0j * _f * _op
                    },
                ]
            else:
                ans = [
                    {
                        _i_end: zeros
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
        SparseSPO(tdse_list + heom_list,
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
    if parameters['visualize_frame'] == True:
        from tenso.libs.drawing import visualize_frame
        visualize_frame(frame, fname=fname)
        print(f"Frame graph is saved as {fname}.pdf", flush=True)
    link_it = frame.node_link_visitor(root)
    tracking_dims = []
    tracking_info = []
    for p, i, q, j in link_it:
        if (q, j) not in tracking_dims:
            tracking_dims.append((p, i))
            tracking_info.append(f"{p}-{q}")
    logger1 = Logger(filename=fname + '.dat.log', level='info')
    logger1.info('# time rdo00 rdo01 rdo10 rdo11')
    logger2 = Logger(filename=fname + '.debug.log', level='info')
    logger2.info('# ' + propagator.info())
    logger2.info(f'# {frame}')
    logger2.info('# time ode_steps max_rank tr')
    logger2.info('# # ' + " ".join(tracking_info))

    renormalize = parameters['renormalize']
    for _t, _s in prop_it:
        time = value(_t, 'time')
        rdo = opt_to_numpy(hierachy.get_densities(state)[hierachy.sys_ends[0]])
        # print(rdo, flush=True)
        rdo *= renorm_coeff
        root_array = state[root]
        ranks = [state.dimension(p, i) for p, i in tracking_dims]
        flat = rdo.reshape(-1)
        trace = (np.trace(rdo)).real

        logger1.info(f'{time} ' + " ".join(f'{p_i:.8f}' for p_i in flat))
        logger2.info(
            f'{time} {propagator.ode_step_counter} {max(ranks) if ranks else 0} {trace}'
        )
        if ranks:
            logger2.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))

        if renormalize:
            norm = np.sqrt(trace)
            state.update({root: root_array / norm})
            renorm_coeff *= norm

        yield time

    if parameters['save_checkpoint_to_file']:
        if parameters['renormalize']:
            state.update({root: root_array * renorm_coeff})
        state.save(fname + default_extension['checkpoint'])
    return


def spin_boson_bath_q(
    fname: str,
    # System
    init_wfn: VecList,
    sys_ham: MatList,
    sys_op: MatList,
    # Bath
    bath: StarBosons,
    # Time-dependent fields: TBD
    td_f: Callable[[float], float] | None = None,
    td_op: MatList | None = None,
    # others
    **kwargs,
) -> Generator[float, None, None]:
    """
    Spin-boson model with MCTDH.
    
    Parameters
    ----------
    fname : str
        The filename prefix for the output files.
    init_wfn : VecList
        The initial wavefunction.
    h : MatList
        The system Hamiltonian.
    op : MatList
        The observable.
    boson_bath : StarBosons
        The boson bath.
    td_f : Callable[[float], float], optional
        The time-dependent field, by default None.
    td_op : MatList, optional
        The time-dependent operator, by default None.
    **kwargs
        The parameters for the simulation.

    Yields
    ------
    float
        Current time in unit in `default_parameters.py`.
    """
    for k, v in kwargs.items():
        if k in parameters:
            parameters[k] = v
        else:
            print(f'Warning: {k} is not a valid parameter; ignored',
                  flush=True)
    print(parameters, flush=True)

    # Frame settings:
    frame_method = parameters['frame_method']
    htd = FrameFactory(1, [bath.k_max])
    if frame_method.lower() == 'train':
        frame, root = htd.train()
    elif frame_method.lower().startswith('tree'):
        n_ary = int(frame_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif frame_method.lower() == 'naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No htd_method {frame_method}.')

    # Boson bath settings:
    dim = parameters['dim']
    if isinstance(dim, int):
        bath_dims = [dim] * bath.k_max
    elif isinstance(dim, list):
        assert len(dim) == bath.k_max
        assert all(isinstance(d, int) for d in dim)
        bath_dims = list(dim)
    else:
        raise NotImplementedError(f'Not dim type {type(dim)}.')

    hierachy = Hierachy(frame, root, htd.sys_ends, htd.bath_ends, [2],
                        [bath_dims])

    # Connect the system part and the bath branch
    ue = quantity(1.0, 'energy')
    print('System H:\n', sys_ham, flush=True)
    rank = parameters['rank']
    if parameters['load_checkpoint_from_file']:
        state = Model.load(fname + default_extension['checkpoint'])
        renorm_coeff = state[root].norm()
        state.update({root: state[root] / renorm_coeff})
    else:
        state = hierachy.initialize_pure_state([init_wfn],
                                               rank=rank,
                                               local_hs=[sys_ham * ue])
        renorm_coeff = 1.0
    tdse_list = hierachy.tdse_list(sys_hamiltonians=[sys_ham * ue],
                                   sys_couplings=[])
    heom_list = hierachy.heom_list([{0: sys_op}], [bath])
    q_list = hierachy.bath_q_list([bath])
    q2_list = hierachy.bath_q2_list([bath])

    if td_f is not None and td_op is not None:
        _op = opt_array(td_op)
        zeros = opt_array(np.zeros_like(td_op))

        def f_list(time: float) -> list[dict[End, OptArray]]:
            _f = td_f(time)
            _i_end = hierachy.sys_ends[0]
            if abs(_f) > 1e-14:
                ans = [
                    {
                        _i_end: -1.0j * _f * _op
                    },
                ]
            else:
                ans = [
                    {
                        _i_end: zeros
                    },
                ]
            return ans
    else:
        f_list = None

    # Propagator:
    start_time = quantity(parameters['start_time'], 'time')
    hamiltonian = SparseSPO(tdse_list + heom_list,
                            f_list=f_list,
                            initial_time=start_time)
    bath_q_op = SparseSPO(q_list)
    bath_q2_op = SparseSPO(q2_list)
    bath_q_state_operation = SPOKet(bath_q_op, state, frame, root)  # q|0>
    bath_q2_state_operation = SPOKet(bath_q2_op, state, frame, root)  # q^2|0>
    # print('<q>:', bath_q_state_operation.close_with_bra(), flush=True)
    # print('<0|q|0>',
    #       SparseInnerProduct(frame, root, state, state, bath_q_op).forward(),
    #       flush=True)
    # print('<q^2>:', bath_q2_state_operation.close_with_bra(), flush=True)
    # print('<0|q^2|0>',
    #       SparseInnerProduct(frame, root, state, state, bath_q2_op).forward(),
    #       flush=True)

    bath_q_state_operation.canonicalize()
    bath_q2_state_operation.canonicalize()
    print('use canonicalize:', flush=True)
    print('<0|q^2|0>:',
          bath_q2_state_operation.close_with_bra() / ue,
          flush=True)
    print('<0|q q|0>',
          ListModelInnerProduct(
              frame, root, bath_q_state_operation.state_list).forward() / ue,
          flush=True)
    # return
    # print('use canonicalize:', flush=True)
    # print('<q>:', bath_q_state_operation.close_with_bra(), flush=True)
    # print('<0|q|0>',
    #       SparseInnerProduct(frame, root, state, state, bath_q_op).forward(),
    #       flush=True)
    # print('<q^2>:', bath_q2_state_operation.close_with_bra(), flush=True)
    # print('<0|q^2|0>',
    #       SparseInnerProduct(frame, root, state, state, bath_q2_op).forward(),
    #       flush=True)

    sp_kwargs = {
        k: v
        for k in SparsePropagator.keyword_settings
        if (v := parameters.get(k)) is not None
    }
    SparsePropagator.update_settings(**sp_kwargs)
    state_propagator = SparsePropagator(hamiltonian, state, frame, root)
    bath_q_state_propagators = [
        SparsePropagator(hamiltonian, _s, frame, root)
        for _s in bath_q_state_operation.state_list
    ]

    #propagation_method = parameters['stepwise_method']
    #ps_method = parameters['ps_method']
    # Use PS1 only for now
    end = quantity(parameters['end_time'], 'time')
    dt = quantity(parameters['step_time'], 'time')
    all_propagators = [state_propagator] + bath_q_state_propagators
    # if propagation_method == 'simple':
    #     _it = _p.propagate(end, dt, ps_method)
    # elif propagation_method == 'mix':
    #     if (dt1 := parameters['auxiliary_step_time']) is not None:
    #         dt1 = quantity(dt1, 'time')
    #     _it = state_propagator.mixed_propagate(
    #         end,
    #         dt,
    #         ending_ps_method=ps_method,
    #         starting_dt=dt1,
    #         starting_ps_method=parameters['auxiliary_ps_method'],
    #         max_starting_rank=parameters['max_auxiliary_rank'],
    #         max_starting_steps=parameters['max_auxiliary_steps'],
    #     )
    # else:
    #     raise NotImplementedError(
    #         f'No propagation method {propagation_method}.')

    if parameters['visualize_frame'] == True:
        from tenso.libs.drawing import visualize_frame
        visualize_frame(frame, fname=fname)
        print(f"Frame graph is saved as {fname}.pdf", flush=True)
    # Output logger:
    link_it = frame.node_link_visitor(root)
    tracking_dims = []
    tracking_info = []
    for p, i, q, j in link_it:
        if (q, j) not in tracking_dims:
            tracking_dims.append((p, i))
            tracking_info.append(f"{p}-{q}")
    logger1 = Logger(filename=fname + '.dat.log', level='info')
    logger1.info('# time rdo00 rdo01 rdo10 rdo11')
    logger2 = Logger(filename=fname + '.debug.log', level='info')
    logger2.info('# ' + state_propagator.info())
    logger2.info(f'# {frame}')
    logger2.info('# time ode_steps max_rank tr')
    logger2.info('# # ' + " ".join(tracking_info))
    logger3 = Logger(filename=fname + '.bath.log', level='info')
    logger3.info('# time <q(t)q(0)>')

    renormalize = parameters['renormalize']
    t = 0.0
    for _n in range(ceil(end / dt)):
        _qs_list = [p.state for p in bath_q_state_propagators]
        _s = state_propagator.state
        time = value(t, 'time')
        rdo = opt_to_numpy(hierachy.get_densities(state)[hierachy.sys_ends[0]])
        # print(rdo, flush=True)
        rdo *= renorm_coeff
        root_array = state[root]
        ranks = [state.dimension(p, i) for p, i in tracking_dims]
        flat = rdo.reshape(-1)
        trace = (np.trace(rdo)).real

        logger1.info(f'{time} ' + " ".join(f'{p_i:.8f}' for p_i in flat))
        logger2.info(
            f'{time} {state_propagator.ode_step_counter} {max(ranks) if ranks else 0} {trace}'
        )

        qs_operation = SPOKet(bath_q_op, _s, frame, root)
        ip = ListModelInnerProduct(frame, root, qs_operation.state_list,
                                   _qs_list).forward()

        logger3.info(f'{time}  {ip / ue}')

        if ranks:
            logger2.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))
            # logger2.info('# ' + ' '.join([f'{_r:d}' for _r in q_ranks]))

        if renormalize:
            norm = np.sqrt(trace)
            state.update({root: root_array / norm})
            renorm_coeff *= norm

        yield time

        # Manually propagate all states
        for _p in all_propagators:
            _p.vmf_step(dt)
        t += dt

    if parameters['save_checkpoint_to_file']:
        if parameters['renormalize']:
            state.update({root: root_array * renorm_coeff})
        state.save(fname + default_extension['checkpoint'])
    return


def spin_boson_bath_q_correlation(
    fname: str,
    # System
    init_wfn: VecList,
    sys_ham: MatList,
    sys_op: MatList,
    # Bath
    bath: StarBosons,
    # Time-dependent fields: TBD
    td_f: Callable[[float], float] | None = None,
    td_op: MatList | None = None,
    # others
    **kwargs,
) -> Generator[float, None, None]:
    raise NotImplementedError("This function is not implemented yet.")
    """
    Spin-boson model with MCTDH.
    
    Parameters
    ----------
    fname : str
        The filename prefix for the output files.
    init_wfn : VecList
        The initial wavefunction.
    h : MatList
        The system Hamiltonian.
    op : MatList
        The observable.
    boson_bath : StarBosons
        The boson bath.
    td_f : Callable[[float], float], optional
        The time-dependent field, by default None.
    td_op : MatList, optional
        The time-dependent operator, by default None.
    **kwargs
        The parameters for the simulation.

    Yields
    ------
    float
        Current time in unit in `default_parameters.py`.
    """
    for k, v in kwargs.items():
        if k in parameters:
            parameters[k] = v
        else:
            print(f'Warning: {k} is not a valid parameter; ignored',
                  flush=True)
    print(parameters, flush=True)

    # Frame settings:
    frame_method = parameters['frame_method']
    htd = FrameFactory(1, [bath.k_max])
    if frame_method.lower() == 'train':
        frame, root = htd.train()
    elif frame_method.lower().startswith('tree'):
        n_ary = int(frame_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif frame_method.lower() == 'naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No htd_method {frame_method}.')

    # Boson bath settings:
    dim = parameters['dim']
    if isinstance(dim, int):
        bath_dims = [dim] * bath.k_max
    elif isinstance(dim, list):
        assert len(dim) == bath.k_max
        assert all(isinstance(d, int) for d in dim)
        bath_dims = list(dim)
    else:
        raise NotImplementedError(f'Not dim type {type(dim)}.')

    hierachy = Hierachy(frame, root, htd.sys_ends, htd.bath_ends, [2],
                        [bath_dims])

    # Connect the system part and the bath branch
    ue = quantity(1.0, 'energy')
    print('System H:\n', sys_ham, flush=True)
    rank = parameters['rank']
    if parameters['load_checkpoint_from_file']:
        state = Model.load(fname + default_extension['checkpoint'])
        renorm_coeff = state[root].norm()
        state.update({root: state[root] / renorm_coeff})
    else:
        state = hierachy.initialize_pure_state([init_wfn],
                                               rank=rank,
                                               local_hs=[sys_ham * ue])
        renorm_coeff = 1.0
    tdse_list = hierachy.tdse_list(sys_hamiltonians=[sys_ham * ue],
                                   sys_couplings=[])
    heom_list = hierachy.heom_list([{0: sys_op}], [bath])
    q_list = hierachy.bath_q_list([bath])

    if td_f is not None and td_op is not None:
        _op = opt_array(td_op)
        zeros = opt_array(np.zeros_like(td_op))

        def f_list(time: float) -> list[dict[End, OptArray]]:
            _f = td_f(time) * ue
            _i_end = hierachy.sys_ends[0]
            if abs(_f) > 1e-14:
                ans = [
                    {
                        _i_end: -1.0j * _f * _op
                    },
                ]
            else:
                ans = [
                    {
                        _i_end: zeros
                    },
                ]
            return ans
    else:
        f_list = None

    # Propagator:
    propagator.SparsePropagator(vmf_atol=parameters['vmf_atol'],
                                ps2_atol=parameters['ps2_atol'],
                                ps2_ratio=parameters['ps2_ratio'],
                                ode_atol=parameters['ode_atol'],
                                ode_rtol=parameters['ode_rtol'],
                                ode_method=parameters['ode_method'])
    start_time = quantity(parameters['start_time'], 'time')
    propagator = SparsePropagator(
        SparseSPO(tdse_list + heom_list,
                  f_list=f_list,
                  initial_time=start_time), state, frame, root)

    bath_q_op = SparseSPO(q_list)

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
    logger1 = Logger(filename=fname + '.dat.log', level='info')
    logger1.info('# time rdo00 rdo01 rdo10 rdo11')
    logger2 = Logger(filename=fname + '.debug.log', level='info')
    logger2.info('# ' + propagator.info())
    logger2.info(f'# {frame}')
    logger2.info('# time ode_steps max_rank tr')
    logger2.info('# # ' + " ".join(tracking_info))
    logger3 = Logger(filename=fname + '.bath.log', level='info')
    logger3.info('# time bath_q')

    renormalize = parameters['renormalize']
    for _t, _s in prop_it:
        time = value(_t, 'time')
        rdo = opt_to_numpy(hierachy.get_densities(state)[hierachy.sys_ends[0]])
        # print(rdo, flush=True)
        rdo *= renorm_coeff
        root_array = state[root]
        ranks = [state.dimension(p, i) for p, i in tracking_dims]
        flat = rdo.reshape(-1)
        trace = (np.trace(rdo)).real

        logger1.info(f'{time} ' + " ".join(f'{p_i:.8f}' for p_i in flat))
        logger2.info(
            f'{time} {propagator.ode_step_counter} {max(ranks) if ranks else 0} {trace}'
        )

        qc = SparseExpectation(bath_q_op, _s, frame, root).forward()
        logger3.info(f'{time} {qc}')

        if ranks:
            logger2.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))

        if renormalize:
            norm = np.sqrt(trace)
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
    init_wfn: VecList,
    sys_ham: MatList,
    sys_ops: list[MatList],
    # Bath
    baths: list[StarBosons],
    # Time-dependent fields: TBD
    td_f: Callable[[float], float] | None = None,
    td_op: MatList | None = None,
    # others
    **kwargs,
) -> Generator[float, None, None]:
    """
    Spin-boson-model-like model with one system DOF and multiple baths with MCTDH.
    
    Parameters
    ----------
    fname : str
        The filename prefix for the output files.
    init_wfn : VecList
        The initial wavefunction.
    h : MatList
        The system Hamiltonian.
    op : MatList
        The observable.
    boson_bath : StarBosons
        The boson bath.
    td_f : Callable[[float], float], optional
        The time-dependent field, by default None.
    td_op : MatList, optional
        The time-dependent operator, by default None.
    **kwargs
        The parameters for the simulation.

    Yields
    ------
    float
        Current time in unit in `default_parameters.py`.
    """
    for k, v in kwargs.items():
        if k in parameters:
            parameters[k] = v
        else:
            print(f'Warning: {k} is not a valid parameter; ignored',
                  flush=True)
    print(parameters, flush=True)

    # Frame settings:
    frame_method = parameters['frame_method']
    htd = FrameFactory(1, [bath.k_max for bath in baths])
    if frame_method.lower() == 'train':
        frame, root = htd.train()
    elif frame_method.lower().startswith('tree'):
        n_ary = int(frame_method[4:])
        frame, root = htd.tree(n_ary=n_ary)
    elif frame_method.lower() == 'naive':
        frame, root = htd.naive()
    else:
        raise NotImplementedError(f'No htd_method {frame_method}.')

    # Boson bath settings:
    sys_dim = len(init_wfn)
    dim = parameters['dim']
    bath_dims = []  # type: list[list[int]]
    for _n, bath in enumerate(baths):
        if isinstance(dim, int):
            bath_dims.append([dim] * bath.k_max)
        elif isinstance(dim, list):
            assert len(dim) == len(baths)
            d_n = dim[_n]
            assert isinstance(d_n, list) and len(d_n) == bath.k_max
            assert all(isinstance(d, int) for d in d_n)
            bath_dims.append(list(d_n))
        else:
            raise TypeError(f'Dim format {dim} is not valid.')

    hierachy = Hierachy(frame, root, htd.sys_ends, htd.bath_ends, [sys_dim],
                        bath_dims)

    # Connect the system part and the bath branch
    ue = quantity(1.0, 'energy')
    print(sys_ham, flush=True)
    rank = parameters['rank']
    if parameters['load_checkpoint_from_file']:
        state = Model.load(fname + default_extension['checkpoint'])
        renorm_coeff = state[root].norm()
        state.update({root: state[root] / renorm_coeff})
    else:
        state = hierachy.initialize_pure_state([init_wfn],
                                               rank=rank,
                                               local_hs=[sys_ham * ue])
        renorm_coeff = 1.0
    tdse_list = hierachy.tdse_list(sys_hamiltonians=[sys_ham * ue],
                                   sys_couplings=[])

    heom_list = hierachy.heom_list([{0: op} for op in sys_ops], baths)

    if td_f is not None and td_op is not None:
        _op = opt_array(td_op)
        zeros = opt_array(np.zeros_like(td_op))

        def f_list(time: float) -> list[dict[End, OptArray]]:
            _f = td_f(time)
            _i_end = hierachy.sys_ends[0]
            if abs(_f) > 1e-14:
                ans = [
                    {
                        _i_end: -1.0j * _f * _op
                    },
                ]
            else:
                ans = [
                    {
                        _i_end: zeros
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
        SparseSPO(tdse_list + heom_list,
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
    logger1 = Logger(filename=fname + '.dat.log', level='info')
    logger1.info('# time rdo00 rdo01 rdo10 rdo11')
    logger2 = Logger(filename=fname + '.debug.log', level='info')
    logger2.info('# ' + propagator.info())
    logger2.info(f'# {frame}')
    logger2.info('# time ode_steps max_rank tr')
    logger2.info('# # ' + " ".join(tracking_info))

    renormalize = parameters['renormalize']
    for _t, _s in prop_it:
        time = value(_t, 'time')
        rdo = opt_to_numpy(hierachy.get_densities(state)[hierachy.sys_ends[0]])
        # print(rdo, flush=True)
        rdo *= renorm_coeff
        root_array = state[root]
        ranks = [state.dimension(p, i) for p, i in tracking_dims]
        flat = rdo.reshape(-1)
        trace = (np.trace(rdo)).real

        logger1.info(f'{time} ' + " ".join(f'{p_i:.8f}' for p_i in flat))
        logger2.info(
            f'{time} {propagator.ode_step_counter} {max(ranks) if ranks else 0} {trace}'
        )
        if ranks:
            logger2.info('# ' + ' '.join([f'{_r:d}' for _r in ranks]))

        if renormalize:
            norm = np.sqrt(trace)
            state.update({root: root_array / norm})
            renorm_coeff *= norm

        yield time

    if parameters['save_checkpoint_to_file']:
        if parameters['renormalize']:
            state.update({root: root_array * renorm_coeff})
        state.save(fname + default_extension['checkpoint'])
    return
