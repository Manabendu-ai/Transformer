<div align = "center">

# Tranformer 

<p> <i> Building a tranformer model from scratch using <b>Pytorch</b> </i> </p>

</div>


## Project Setup
```shell
cd transformer/
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Transformer Architecture
```mermaid
flowchart BT
    %% ENCODER
    subgraph ENC["Encoder (Nx)"]
        direction BT
        EMB_IN["Input<br/>Embedding"]
        MHA_E["Multi-Head<br/>Attention"]
        AN1_E["Add & Norm"]
        FF_E["Feed<br/>Forward"]
        AN2_E["Add & Norm"]

        EMB_IN --> MHA_E --> AN1_E --> FF_E --> AN2_E
        EMB_IN -.->|residual| AN1_E
        AN1_E -.->|residual| AN2_E
    end

    INPUTS["Inputs"] --> EMB_IN
    PE_IN(("Positional<br/>Encoding")) --> EMB_IN

    %% DECODER
    subgraph DEC["Decoder (Nx)"]
        direction BT
        EMB_OUT["Output<br/>Embedding"]
        MMHA["Masked<br/>Multi-Head<br/>Attention"]
        AN1_D["Add & Norm"]
        MHA_D["Multi-Head<br/>Attention"]
        AN2_D["Add & Norm"]
        FF_D["Feed<br/>Forward"]
        AN3_D["Add & Norm"]

        EMB_OUT --> MMHA --> AN1_D --> MHA_D --> AN2_D --> FF_D --> AN3_D
        EMB_OUT -.->|residual| AN1_D
        AN1_D -.->|residual| AN2_D
        AN2_D -.->|residual| AN3_D
    end

    OUTPUTS["Outputs<br/>(shifted right)"] --> EMB_OUT
    PE_OUT(("Positional<br/>Encoding")) --> EMB_OUT

    %% Cross-attention: encoder output feeds decoder's second attention block
    AN2_E ==>|"K, V"| MHA_D

    %% OUTPUT HEAD
    AN3_D --> LIN["Linear"] --> SM["Softmax"] --> PROB["Output<br/>Probabilities"]

    %% STYLES
    classDef embed fill:#fadadd,stroke:#333,color:#000
    classDef attn fill:#fde0b8,stroke:#333,color:#000
    classDef ff fill:#bfe3f5,stroke:#333,color:#000
    classDef norm fill:#f3f5b5,stroke:#333,color:#000
    classDef lin fill:#dcdcf5,stroke:#333,color:#000
    classDef soft fill:#cfe8cf,stroke:#333,color:#000
    classDef io fill:#fff,stroke:none,color:#000

    class EMB_IN,EMB_OUT embed
    class MHA_E,MMHA,MHA_D attn
    class FF_E,FF_D ff
    class AN1_E,AN2_E,AN1_D,AN2_D,AN3_D norm
    class LIN lin
    class SM soft
    class INPUTS,OUTPUTS,PROB io
```
