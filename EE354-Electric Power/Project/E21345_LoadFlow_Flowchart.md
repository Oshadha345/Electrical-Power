# Flowchart for `E21345_LoadFlow.py` (Task-1)

Student Name: **Samarakoon S.M.O.T.**  
Student ID: **E/21/345**

This flowchart is prepared to match the assignment requirement:
- Each flowchart box includes the related code line numbers.
- The program flow shown here is only for **Task-1** (custom Full Newton-Raphson load flow).

## 1) Main Program Flow (with line references)

```mermaid
flowchart TD
    A([Start Program<br/>L574-L575]) --> B[Parse CLI args<br/>parse_cli_args()<br/>L508-L535, L544]
    B --> C[Resolve JSON path and check file exists<br/>L545-L550]
    C --> D[Print execution proof banner<br/>print_banner()<br/>L383-L396, L552]
    D --> E[Load JSON data<br/>load_json_data()<br/>L51-L58, L554]
    E --> F[Build system model<br/>make_system_model()<br/>L61-L145, L555]
    F --> G[Construct Y-bus matrix<br/>build_ybus()<br/>L148-L175, L556]
    G --> H[Run Full Newton-Raphson solver<br/>run_newton_raphson()<br/>L258-L337, L558]
    H --> I{Converged? <br/>L559-L563}
    I -- No --> J[[Raise RuntimeError<br/>L560-L563]]
    I -- Yes --> K[Print convergence trace<br/>print_iteration_history()<br/>L399-L411, L565-L566]
    K --> L[Print 2nd iteration sample output<br/>print_second_iteration()<br/>L414-L446, L567]
    L --> M[Print final bus voltages and injections<br/>print_final_bus_results()<br/>L449-L478, L568]
    M --> N[Compute line/transformer flows and losses<br/>compute_branch_flows()<br/>L340-L380, L570]
    N --> O[Print branch results and total losses<br/>print_branch_results()<br/>L481-L505, L571]
    O --> P([End Program])
```

## 2) Newton-Raphson Internal Loop Detail (with line references)

```mermaid
flowchart TD
    A1[Setup bus indexes and state vectors<br/>L269-L283] --> A2[Flat start initialization<br/>v=1.0 pu, angle=0<br/>L275-L280]
    A2 --> A3[Iteration loop begin<br/>for iteration in range()<br/>L285]
    A3 --> A4[Calculate P and Q injections<br/>calc_power_injections()<br/>L286, L178-L187]
    A4 --> A5[Build mismatch vector ΔP, ΔQ<br/>L288-L291]
    A5 --> A6[Store iteration history<br/>L293-L304]
    A6 --> A7{Max mismatch < tolerance?<br/>L306-L308}
    A7 -- Yes --> A8[Exit loop as converged<br/>L307-L308]
    A7 -- No --> A9[Build Jacobian J1-J4<br/>build_jacobian()<br/>L310, L190-L255]
    A9 --> A10[Solve linear system J·Δx = mismatch<br/>L311]
    A10 --> A11[Update angles and PQ voltages<br/>L313-L318]
    A11 --> A12[Re-apply fixed slack/PV voltages<br/>L320-L323]
    A12 --> A3
    A8 --> A13[Final P,Q calc and return result dict<br/>L325-L337]
```

## 3) Box-to-Code Mapping Table

| Flowchart Box | Action | Function / Section | Code Lines |
|---|---|---|---|
| Main-B | Read CLI options (`--data`, `--tol`, `--max-iter`) | `parse_cli_args` | `L508-L535` |
| Main-C | Input file path handling and existence check | `main` | `L545-L550` |
| Main-D | Print name, ID, timestamp for execution proof | `print_banner` | `L383-L396`, call at `L552` |
| Main-E | Read JSON file | `load_json_data` | `L51-L58` |
| Main-F | Build bus sets, power specs, branch records | `make_system_model` | `L61-L145` |
| Main-G | Create Y-bus from branch data | `build_ybus` | `L148-L175` |
| Main-H | Newton-Raphson solve | `run_newton_raphson` | `L258-L337` |
| Main-I/J | Convergence check and fail path | `main` | `L559-L563` |
| Main-K | Print mismatch by iteration | `print_iteration_history` | `L399-L411` |
| Main-L | Print assignment-required 2nd iteration results | `print_second_iteration` | `L414-L446` |
| Main-M | Print final bus results | `print_final_bus_results` | `L449-L478` |
| Main-N | Compute branch flows and system losses | `compute_branch_flows` | `L340-L380` |
| Main-O | Print line/transformer flows and total losses | `print_branch_results` | `L481-L505` |
| NR-A9 | Build Jacobian submatrices J1, J2, J3, J4 | `build_jacobian` | `L190-L255` |

## 4) How to Use in Report

1. Keep this markdown file as a traceability reference.
2. Export the Mermaid diagram as image (PNG/SVG) and insert into your report.
3. Mention that each block is linked to exact source-code lines in `E21345_LoadFlow.py`.

