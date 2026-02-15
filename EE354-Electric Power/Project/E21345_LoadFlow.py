"""
E21345_LoadFlow.py
Full Newton-Raphson load flow solver for IEEE 9-bus system (Task-1).

Student Name: Samarakoon S.M.O.T.
Student ID  : E/21/345
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


STUDENT_NAME = "Samarakoon S.M.O.T."
STUDENT_ID = "E/21/345"
DEFAULT_BASE_MVA = 100.0
DEFAULT_TOL = 1e-4
DEFAULT_MAX_ITER = 30


@dataclass
class SystemModel:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Keeps all processed system data in one place.
    """

    base_mva: float
    bus_numbers: List[int]
    bus_index: Dict[int, int]
    index_bus: Dict[int, int]
    slack_bus: int
    pv_buses: List[int]
    pq_buses: List[int]
    p_spec: np.ndarray
    q_spec: np.ndarray
    p_gen: np.ndarray
    p_load: np.ndarray
    q_load: np.ndarray
    v_set: np.ndarray
    branches: List[dict]


def load_json_data(json_path: Path) -> dict:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Reads the json file and returns parsed python data.
    """

    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_system_model(raw_data: dict, base_mva: float, slack_bus: int = 1) -> SystemModel:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Builds bus sets and per-unit power specs from input data.
    """

    buses = raw_data["bus_data"]["buses"]
    generators = raw_data["generator_data"]["generators"]
    transformers = raw_data["transformer_data"]["transformers"]
    lines = raw_data["transmission_line_data"]["lines"]
    loads = raw_data["load_data"]["loads"]

    bus_numbers = sorted(bus["bus_number"] for bus in buses)
    n_bus = len(bus_numbers)
    bus_index = {bus: idx for idx, bus in enumerate(bus_numbers)}
    index_bus = {idx: bus for bus, idx in bus_index.items()}

    generator_buses = sorted(g["bus_number"] for g in generators)
    pv_buses = [bus for bus in generator_buses if bus != slack_bus]
    pq_buses = [bus for bus in bus_numbers if bus not in ([slack_bus] + pv_buses)]

    p_gen = np.zeros(n_bus, dtype=float)
    p_load = np.zeros(n_bus, dtype=float)
    q_load = np.zeros(n_bus, dtype=float)
    v_set = np.ones(n_bus, dtype=float)

    for gen in generators:
        idx = bus_index[gen["bus_number"]]
        p_gen[idx] = gen["scheduled_active_power_MW"] / base_mva
        v_set[idx] = gen["scheduled_voltage_pu"]

    for load in loads:
        idx = bus_index[load["bus_number"]]
        p_load[idx] = load["P_MW"] / base_mva
        q_load[idx] = load["Q_MVAr"] / base_mva

    # P_spec = generation - load, Q_spec only known at PQ loads.
    p_spec = p_gen - p_load
    q_spec = -q_load

    branches: List[dict] = []
    for line in lines:
        branches.append(
            {
                "kind": "Line",
                "from_bus": line["from_bus"],
                "to_bus": line["to_bus"],
                "r_pu": line["R_pu"],
                "x_pu": line["X_pu"],
                "g_pu": line["G_pu"],
                "b_pu": line["B_pu"],
            }
        )

    for tr in transformers:
        branches.append(
            {
                "kind": "Transformer",
                "from_bus": tr["from_bus"],
                "to_bus": tr["to_bus"],
                "r_pu": tr["R_pu"],
                "x_pu": tr["X_pu"],
                "g_pu": tr["G_pu"],
                "b_pu": tr["B_pu"],
            }
        )

    branches.sort(key=lambda row: (row["from_bus"], row["to_bus"], row["kind"]))

    return SystemModel(
        base_mva=base_mva,
        bus_numbers=bus_numbers,
        bus_index=bus_index,
        index_bus=index_bus,
        slack_bus=slack_bus,
        pv_buses=pv_buses,
        pq_buses=pq_buses,
        p_spec=p_spec,
        q_spec=q_spec,
        p_gen=p_gen,
        p_load=p_load,
        q_load=q_load,
        v_set=v_set,
        branches=branches,
    )


def build_ybus(system: SystemModel) -> np.ndarray:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Creates Y-bus matrix from line and transformer records.
    """

    n_bus = len(system.bus_numbers)
    y_bus = np.zeros((n_bus, n_bus), dtype=complex)

    for branch in system.branches:
        i = system.bus_index[branch["from_bus"]]
        j = system.bus_index[branch["to_bus"]]

        z = complex(branch["r_pu"], branch["x_pu"])
        if abs(z) < 1e-12:
            raise ValueError(
                f"Branch impedance too small between bus {branch['from_bus']} and {branch['to_bus']}."
            )

        y_series = 1.0 / z
        y_shunt_half = complex(branch["g_pu"], branch["b_pu"]) / 2.0

        y_bus[i, i] += y_series + y_shunt_half
        y_bus[j, j] += y_series + y_shunt_half
        y_bus[i, j] -= y_series
        y_bus[j, i] -= y_series

    return y_bus


def calc_power_injections(v_mag: np.ndarray, theta: np.ndarray, y_bus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Computes calculated P and Q injections at all buses.
    """

    voltage = v_mag * np.exp(1j * theta)
    current = y_bus @ voltage
    power = voltage * np.conj(current)
    return power.real, power.imag


def build_jacobian(
    v_mag: np.ndarray,
    theta: np.ndarray,
    y_bus: np.ndarray,
    angle_idx: List[int],
    pq_idx: List[int],
    p_calc: np.ndarray,
    q_calc: np.ndarray,
) -> np.ndarray:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Builds full Jacobian matrix (J1, J2, J3, J4).
    """

    g_mat = y_bus.real
    b_mat = y_bus.imag

    n_theta = len(angle_idx)
    n_pq = len(pq_idx)

    j1 = np.zeros((n_theta, n_theta), dtype=float)
    j2 = np.zeros((n_theta, n_pq), dtype=float)
    j3 = np.zeros((n_pq, n_theta), dtype=float)
    j4 = np.zeros((n_pq, n_pq), dtype=float)

    for row, i in enumerate(angle_idx):
        for col, j in enumerate(angle_idx):
            if i == j:
                j1[row, col] = -q_calc[i] - b_mat[i, i] * (v_mag[i] ** 2)
            else:
                delta = theta[i] - theta[j]
                j1[row, col] = v_mag[i] * v_mag[j] * (
                    g_mat[i, j] * np.sin(delta) - b_mat[i, j] * np.cos(delta)
                )

    for row, i in enumerate(angle_idx):
        for col, j in enumerate(pq_idx):
            if i == j:
                j2[row, col] = (p_calc[i] / v_mag[i]) + g_mat[i, i] * v_mag[i]
            else:
                delta = theta[i] - theta[j]
                j2[row, col] = v_mag[i] * (
                    g_mat[i, j] * np.cos(delta) + b_mat[i, j] * np.sin(delta)
                )

    for row, i in enumerate(pq_idx):
        for col, j in enumerate(angle_idx):
            if i == j:
                j3[row, col] = p_calc[i] - g_mat[i, i] * (v_mag[i] ** 2)
            else:
                delta = theta[i] - theta[j]
                j3[row, col] = -v_mag[i] * v_mag[j] * (
                    g_mat[i, j] * np.cos(delta) + b_mat[i, j] * np.sin(delta)
                )

    for row, i in enumerate(pq_idx):
        for col, j in enumerate(pq_idx):
            if i == j:
                j4[row, col] = (q_calc[i] / v_mag[i]) - b_mat[i, i] * v_mag[i]
            else:
                delta = theta[i] - theta[j]
                j4[row, col] = v_mag[i] * (
                    g_mat[i, j] * np.sin(delta) - b_mat[i, j] * np.cos(delta)
                )

    return np.block([[j1, j2], [j3, j4]])


def run_newton_raphson(
    system: SystemModel,
    y_bus: np.ndarray,
    tol: float,
    max_iter: int,
) -> dict:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Solves load flow using full Newton-Raphson iterations.
    """

    n_bus = len(system.bus_numbers)
    slack_idx = system.bus_index[system.slack_bus]
    pv_idx = [system.bus_index[bus] for bus in system.pv_buses]
    pq_idx = [system.bus_index[bus] for bus in system.pq_buses]
    angle_idx = [idx for idx in range(n_bus) if idx != slack_idx]

    # Flat start for unknown states. Then fixed buses are forced to set values.
    v_mag = np.ones(n_bus, dtype=float)
    theta = np.zeros(n_bus, dtype=float)
    v_mag[slack_idx] = system.v_set[slack_idx]
    if pv_idx:
        v_mag[pv_idx] = system.v_set[pv_idx]

    history: List[dict] = []
    converged = False

    for iteration in range(1, max_iter + 1):
        p_calc, q_calc = calc_power_injections(v_mag, theta, y_bus)

        d_p = system.p_spec[angle_idx] - p_calc[angle_idx]
        d_q = system.q_spec[pq_idx] - q_calc[pq_idx]
        mismatch = np.concatenate([d_p, d_q])
        max_mismatch = float(np.max(np.abs(mismatch))) if mismatch.size > 0 else 0.0

        history.append(
            {
                "iteration": iteration,
                "v_mag": v_mag.copy(),
                "theta": theta.copy(),
                "p_calc": p_calc.copy(),
                "q_calc": q_calc.copy(),
                "d_p": d_p.copy(),
                "d_q": d_q.copy(),
                "max_mismatch": max_mismatch,
            }
        )

        if max_mismatch < tol:
            converged = True
            break

        jac = build_jacobian(v_mag, theta, y_bus, angle_idx, pq_idx, p_calc, q_calc)
        correction = np.linalg.solve(jac, mismatch)

        d_theta = correction[: len(angle_idx)]
        d_v = correction[len(angle_idx) :]

        theta[angle_idx] += d_theta
        if len(pq_idx) > 0:
            v_mag[pq_idx] += d_v

        # keep fixed-voltage buses exactly fixed after each update
        v_mag[slack_idx] = system.v_set[slack_idx]
        if pv_idx:
            v_mag[pv_idx] = system.v_set[pv_idx]

    p_calc, q_calc = calc_power_injections(v_mag, theta, y_bus)

    return {
        "converged": converged,
        "iterations": len(history),
        "v_mag": v_mag,
        "theta": theta,
        "p_calc": p_calc,
        "q_calc": q_calc,
        "history": history,
        "angle_idx": angle_idx,
        "pq_idx": pq_idx,
    }


def compute_branch_flows(system: SystemModel, v_mag: np.ndarray, theta: np.ndarray) -> Tuple[List[dict], complex]:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Computes branch power flows and total system losses.
    """

    voltage = v_mag * np.exp(1j * theta)
    flow_rows: List[dict] = []
    total_loss = 0.0 + 0.0j

    for branch in system.branches:
        i = system.bus_index[branch["from_bus"]]
        j = system.bus_index[branch["to_bus"]]

        y_series = 1.0 / complex(branch["r_pu"], branch["x_pu"])
        y_shunt_half = complex(branch["g_pu"], branch["b_pu"]) / 2.0

        i_ij = (voltage[i] - voltage[j]) * y_series + voltage[i] * y_shunt_half
        i_ji = (voltage[j] - voltage[i]) * y_series + voltage[j] * y_shunt_half

        s_ij = voltage[i] * np.conj(i_ij) * system.base_mva
        s_ji = voltage[j] * np.conj(i_ji) * system.base_mva

        loss = s_ij + s_ji
        total_loss += loss

        flow_rows.append(
            {
                "kind": branch["kind"],
                "from_bus": branch["from_bus"],
                "to_bus": branch["to_bus"],
                "p_from_to": s_ij.real,
                "q_from_to": s_ij.imag,
                "p_to_from": s_ji.real,
                "q_to_from": s_ji.imag,
                "p_loss": loss.real,
                "q_loss": loss.imag,
            }
        )

    return flow_rows, total_loss


def print_banner(json_path: Path, tol: float) -> None:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Prints execution proof details for screenshot.
    """

    print("=" * 88)
    print("EE354 - Full Newton-Raphson Load Flow (Task-1)")
    print(f"Student Name: {STUDENT_NAME}")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input JSON: {json_path}")
    print(f"Convergence Tolerance: {tol:.1e} pu")
    print("=" * 88)


def print_iteration_history(history: List[dict]) -> None:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Prints max mismatch per iteration.
    """

    print("\nConvergence trace (max mismatch):")
    print("-" * 40)
    print(f"{'Iteration':>10} | {'Max mismatch (pu)':>18}")
    print("-" * 40)
    for row in history:
        print(f"{row['iteration']:>10d} | {row['max_mismatch']:>18.8f}")
    print("-" * 40)


def print_second_iteration(system: SystemModel, result: dict) -> None:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Prints second-iteration values for assignment sample output.
    """

    print("\nSample output required by assignment: 2nd iteration results")
    print("-" * 88)

    if len(result["history"]) < 2:
        print("Solver converged before second iteration, so there is no iter-2 snapshot.")
        return

    iter_two = result["history"][1]
    print(f"{'Bus':>5} | {'|V|_iter2 (pu)':>14} | {'Angle_iter2 (deg)':>18} | {'P_calc (pu)':>12} | {'Q_calc (pu)':>12}")
    print("-" * 88)
    for idx, bus in enumerate(system.bus_numbers):
        v_val = iter_two["v_mag"][idx]
        ang_val = np.degrees(iter_two["theta"][idx])
        p_val = iter_two["p_calc"][idx]
        q_val = iter_two["q_calc"][idx]
        print(f"{bus:>5d} | {v_val:>14.6f} | {ang_val:>18.6f} | {p_val:>12.6f} | {q_val:>12.6f}")

    print("\nMismatch vector at iteration-2:")
    print(f"{'Bus':>5} | {'Equation':>10} | {'Mismatch (pu)':>14}")
    print("-" * 40)
    mismatch_bus = [system.index_bus[idx] for idx in result["angle_idx"]] + [
        system.index_bus[idx] for idx in result["pq_idx"]
    ]
    mismatch_type = ["Delta P"] * len(result["angle_idx"]) + ["Delta Q"] * len(result["pq_idx"])
    mismatch_values = np.concatenate([iter_two["d_p"], iter_two["d_q"]])
    for bus, eqn, mis in zip(mismatch_bus, mismatch_type, mismatch_values):
        print(f"{bus:>5d} | {eqn:>10} | {mis:>14.6f}")


def print_final_bus_results(system: SystemModel, result: dict) -> None:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Prints final solved bus voltages and injections.
    """

    print("\nFinal bus voltages and injections:")
    print("-" * 104)
    print(
        f"{'Bus':>5} | {'Type':>7} | {'|V| (pu)':>10} | {'Angle (deg)':>12} | "
        f"{'P_inj (MW)':>12} | {'Q_inj (MVAr)':>13}"
    )
    print("-" * 104)

    for idx, bus in enumerate(system.bus_numbers):
        if bus == system.slack_bus:
            bus_type = "Slack"
        elif bus in system.pv_buses:
            bus_type = "PV"
        else:
            bus_type = "PQ"

        v_val = result["v_mag"][idx]
        ang_val = np.degrees(result["theta"][idx])
        p_mw = result["p_calc"][idx] * system.base_mva
        q_mvar = result["q_calc"][idx] * system.base_mva

        print(f"{bus:>5d} | {bus_type:>7} | {v_val:>10.6f} | {ang_val:>12.6f} | {p_mw:>12.6f} | {q_mvar:>13.6f}")

    print("-" * 104)


def print_branch_results(flow_rows: List[dict], total_loss: complex) -> None:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Prints branch flows and total losses.
    """

    print("\nLine/transformer power flows:")
    print("-" * 128)
    print(
        f"{'Type':>12} | {'From':>4} | {'To':>4} | {'P_from_to':>11} | {'Q_from_to':>11} | "
        f"{'P_to_from':>11} | {'Q_to_from':>11} | {'P_loss':>10} | {'Q_loss':>10}"
    )
    print("-" * 128)

    for row in flow_rows:
        print(
            f"{row['kind']:>12} | {row['from_bus']:>4d} | {row['to_bus']:>4d} | "
            f"{row['p_from_to']:>11.6f} | {row['q_from_to']:>11.6f} | "
            f"{row['p_to_from']:>11.6f} | {row['q_to_from']:>11.6f} | "
            f"{row['p_loss']:>10.6f} | {row['q_loss']:>10.6f}"
        )

    print("-" * 128)
    print(f"Total active power loss  (MW)   : {total_loss.real:.6f}")
    print(f"Total reactive power loss (MVAr): {total_loss.imag:.6f}")


def parse_cli_args() -> argparse.Namespace:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Parses command line options for data path and solver settings.
    """

    parser = argparse.ArgumentParser(
        description="Full Newton-Raphson load flow for IEEE 9-bus (Task-1 submission file)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="IEEE9_Bus_system_data.json",
        help="Path to json input data file.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help="Convergence tolerance in pu mismatch (default: 1e-4).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Maximum Newton-Raphson iterations (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Author: Samarakoon S.M.O.T. (E/21/345)
    Brief : Main entry point. Runs whole Task-1 workflow and prints outputs.
    """

    args = parse_cli_args()
    script_dir = Path(__file__).resolve().parent
    json_path = Path(args.data)
    if not json_path.is_absolute():
        json_path = script_dir / json_path
    if not json_path.exists():
        raise FileNotFoundError(f"Input json file not found: {json_path}")

    print_banner(json_path=json_path, tol=args.tol)

    raw_data = load_json_data(json_path)
    system = make_system_model(raw_data=raw_data, base_mva=DEFAULT_BASE_MVA, slack_bus=1)
    y_bus = build_ybus(system)

    result = run_newton_raphson(system=system, y_bus=y_bus, tol=args.tol, max_iter=args.max_iter)
    if not result["converged"]:
        raise RuntimeError(
            f"Load flow did not converge in {args.max_iter} iterations. Last mismatch: "
            f"{result['history'][-1]['max_mismatch']:.6e} pu"
        )

    print(f"Converged in {result['iterations']} iterations.")
    print_iteration_history(result["history"])
    print_second_iteration(system=system, result=result)
    print_final_bus_results(system=system, result=result)

    flow_rows, total_loss = compute_branch_flows(system=system, v_mag=result["v_mag"], theta=result["theta"])
    print_branch_results(flow_rows=flow_rows, total_loss=total_loss)


if __name__ == "__main__":
    main()

