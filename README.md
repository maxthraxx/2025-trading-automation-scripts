# 2025-trading-automation-scripts
2025-trading-automation-scripts


![Example 1](Figure_1.png)
![Example 2](Figure_2.png)

flowchart TD
    A[OHLC Data Input] --> B[Normalize Close Position]
    B --> C[Compute Beta Distribution Parameters]
    C --> D[Calculate Buying/Selling Pressure]
    
    E[Volume Data Input] -.-> F[Normalize Volume]
    F -.-> C
    
    D --> G[Pressure Direction & Strength]
    D --> H[Statistical Significance Tests]
    D --> I[Momentum Indicators]
    D --> J[Divergence Detection]
    
    G --> K[Final Signal Calculation]
    H --> K
    I --> K
    J --> K
    
    K --> L[Trade Decision Signal -1.0 to 1.0]
    
    subgraph Core Mathematical Components
        B
        C
        D
    end
    
    subgraph Signal Enhancement
        G
        H
        I
        J
    end


