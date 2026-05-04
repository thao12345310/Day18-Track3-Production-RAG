  flowchart TD
    %% ============================================================
    %% STYLES
    %% ============================================================
    classDef entryNode fill:#1a73e8,stroke:#0d47a1,color:#fff,stroke-width:2px
    classDef routingNode fill:#f9a825,stroke:#f57f17,color:#000,stroke-width:2px
    classDef processNode fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef ocrNode fill:#fce4ec,stroke:#c62828,color:#000
    classDef storageNode fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef minioNode fill:#fff3e0,stroke:#e65100,color:#000
    classDef evalNode fill:#f3e5f5,stroke:#6a1b9a,color:#000
    classDef parallelNode fill:#e0f7fa,stroke:#00838f,color:#000
    classDef endNode fill:#424242,stroke:#212121,color:#fff,stroke-width:2px
    classDef fanOutNode fill:#ffe0b2,stroke:#e65100,color:#000,stroke-width:2px
    classDef notifyNode fill:#e8eaf6,stroke:#283593,color:#000
    classDef profileNode fill:#e8eaf6,stroke:#1565c0,color:#000,stroke-width:2px

    %% ============================================================
    %% ENTRY PHASE
    %% ============================================================
    START([📄 POST /ingestion_pipeline/process]):::entryNode
    START --> FTD

    subgraph ENTRY["🔍 Entry Phase"]
        FTD["1. File Type Detection<br/><i>Detect extension → file_type</i>"]:::processNode
        FTD -->|"unknown"| ORPHAN_GATE{Orphan Gate}:::routingNode
        FTD -->|"supported"| LIP

        ORPHAN_GATE -->|"orphan"| OBR["Orphan Bucket Routing<br/><i>Upload to MinIO Orphan bucket</i>"]:::minioNode
        OBR --> END_ORPHAN([🗑️ END — Orphaned]):::endNode

        LIP["2. Load Index Profile<br/><i>GET Knowledge Service → IndexProfile<br/>chunk_strategy, embed_model, db_type</i>"]:::profileNode
        LIP --> INIT["3. Init Storage<br/><i>Neo4j: create Document node<br/>Milvus/Weaviate: ensure collections</i>"]:::storageNode
        INIT --> STL["4. Save to Landing<br/><i>MinIO Landing bucket</i>"]:::minioNode
        STL --> TC["5. Text Check<br/><i>has_text / is_image flags</i>"]:::processNode
    end

    %% ============================================================
    %% TEXT CHECK ROUTING
    %% ============================================================
    TC --> TC_ROUTE{Text Check<br/>Router}:::routingNode
    TC_ROUTE -->|"🖼️ Image / no text"| OCR_NODE
    TC_ROUTE -->|"📄 Document"| FTR

    %% ============================================================
    %% FILE TYPE ROUTER
    %% ============================================================
    FTR["6. File Type Router<br/><i>Route by extension</i>"]:::routingNode
    FTR -->|".doc"| A0_PATH
    FTR -->|".docx"| A1_PATH
    FTR -->|".html / .htm"| A_HTML_PATH
    FTR -->|".md / .markdown"| A2_PATH
    FTR -->|".pdf"| A_PDF_PATH
    FTR -->|".txt .csv .xlsx .xls"| A3_PATH

    %% ============================================================
    %% SUB-PATH A0: DOC → DOCX (joins A1)
    %% ============================================================
    subgraph A0_PATH["🅰️0 DOC Path"]
        A0_CONVERT["Convert DOC → DOCX<br/><i>LibreOffice conversion</i>"]:::processNode
    end
    A0_CONVERT --> A1_SAVE

    %% ============================================================
    %% SUB-PATH A1: DOCX-first (fully sequential)
    %% ============================================================
    subgraph A1_PATH["🅰️1 DOCX Path — Sequential"]
        A1_SAVE["Save DOCX → MinIO<br/><i>Staging bucket</i>"]:::minioNode
        A1_SAVE --> A1_DOCX2HTML["Convert DOCX → HTML"]:::processNode
        A1_DOCX2HTML --> A1_UNROLL["Unrolling Table<br/><i>Expand merged cells</i>"]:::processNode
        A1_UNROLL --> A1_HTML2MD["Convert HTML → MD"]:::processNode
        A1_HTML2MD --> A1_STORE_HTML["Store HTML → Neo4j<br/><i>DR backup</i>"]:::storageNode
    end
    A1_STORE_HTML --> SIX_WAY_ENTRY

    %% ============================================================
    %% SUB-PATH A_HTML: HTML-first
    %% ============================================================
    subgraph A_HTML_PATH["🅰️_HTML Path"]
        A_HTML_ENTRY["A_HTML Entry"]:::processNode
        A_HTML_ENTRY --> A_HTML_EXTRACT["Extract HTML Content<br/><i>Decode base64 → HTML</i>"]:::processNode
        A_HTML_EXTRACT --> AH_UNROLL["Unrolling Table"]:::processNode
        AH_UNROLL --> AH_HTML2MD["Convert HTML → MD"]:::processNode
    end

    AH_HTML2MD --> AH_FAN{A_HTML 2-Way<br/>Fan-Out}:::fanOutNode

    AH_FAN -->|"backup"| AH_STORE_HTML["Store HTML → Neo4j"]:::storageNode
    AH_FAN -->|"docx"| AH_HTML2DOCX["Convert HTML → DOCX"]:::processNode
    AH_HTML2DOCX --> AH_SAVE_DOCX["Save DOCX → MinIO"]:::minioNode

    AH_STORE_HTML --> SIX_WAY_ENTRY
    AH_SAVE_DOCX --> SIX_WAY_ENTRY

    %% ============================================================
    %% SUB-PATH A2: Markdown-first
    %% ============================================================
    subgraph A2_PATH["🅰️2 Markdown Path"]
        A2_ENTRY["A2 Entry"]:::processNode
    end

    A2_ENTRY --> A2_FAN{A2 2-Way<br/>Fan-Out}:::fanOutNode

    A2_FAN -->|"ai (direct)"| SIX_WAY_ENTRY
    A2_FAN -->|"backup artifacts"| A2_MD2HTML["Convert MD → HTML"]:::processNode
    A2_MD2HTML --> A2_STORE_HTML["Store HTML → Neo4j"]:::storageNode
    A2_STORE_HTML --> A2_HTML2DOCX["Convert HTML → DOCX"]:::processNode
    A2_HTML2DOCX --> A2_SAVE_DOCX["Save DOCX → MinIO"]:::minioNode
    A2_SAVE_DOCX --> SIX_WAY_ENTRY

    %% ============================================================
    %% SUB-PATH A_PDF: PDF Layer Detection
    %% ============================================================
    subgraph A_PDF_PATH["🅰️_PDF Path"]
        A_PDF_DETECT["PDF Layer Detection<br/><i>2-layer vs scanned</i>"]:::processNode
    end

    A_PDF_DETECT --> PDF_ROUTE{PDF Layer<br/>Router}:::routingNode
    PDF_ROUTE -->|"2-layer (has text)"| A3_EXTRACT
    PDF_ROUTE -->|"scanned (image-only)"| OCR_NODE

    %% ============================================================
    %% SUB-PATH A3: Generic Text Extraction
    %% ============================================================
    subgraph A3_PATH["🅰️3 Generic Text Path"]
        A3_EXTRACT["Text Extraction<br/><i>txt/csv/xlsx/2-layer PDF</i>"]:::processNode
        A3_EXTRACT --> A3_PRODUCE_HTML["Produce HTML<br/><i>Wrap raw content</i>"]:::processNode
        A3_PRODUCE_HTML --> A3_ENTRY["A3 Entry"]:::processNode
        A3_ENTRY --> A3_UNROLL["Unrolling Table"]:::processNode
        A3_UNROLL --> A3_HTML2MD["Convert HTML → MD"]:::processNode
    end

    A3_HTML2MD --> A3_FAN{A3 2-Way<br/>Fan-Out}:::fanOutNode

    A3_FAN -->|"backup"| A3_STORE_HTML["Store HTML → Neo4j"]:::storageNode
    A3_FAN -->|"docx"| A3_HTML2DOCX["Convert HTML → DOCX"]:::processNode
    A3_HTML2DOCX --> A3_SAVE_DOCX["Save DOCX → MinIO"]:::minioNode

    A3_STORE_HTML --> SIX_WAY_ENTRY
    A3_SAVE_DOCX --> SIX_WAY_ENTRY

    %% ============================================================
    %% JOURNEY B: OCR (Images + Scanned PDFs)
    %% ============================================================
    subgraph JOURNEY_B["🅱️ Journey B — OCR Path"]
        OCR_NODE["OCR via Claude VLM<br/><i>Image/Scanned PDF → HTML</i>"]:::ocrNode
        OCR_NODE --> B_BROKEN_TABLE["Parse Broken Table<br/><i>Fix OCR table artifacts</i>"]:::processNode
        B_BROKEN_TABLE --> B_UNROLL["Unrolling Table"]:::processNode
        B_UNROLL --> B_HTML2MD["Convert HTML → MD"]:::processNode
    end

    B_HTML2MD --> B_FAN{B 2-Way<br/>Fan-Out}:::fanOutNode

    B_FAN -->|"docx"| B_HTML2DOCX["Convert HTML → DOCX"]:::processNode
    B_HTML2DOCX --> B_SAVE_DOCX["Save DOCX → MinIO"]:::minioNode
    B_FAN -->|"backup"| B_STORE_HTML["Store HTML → Neo4j"]:::storageNode

    B_SAVE_DOCX --> SIX_WAY_ENTRY
    B_STORE_HTML --> SIX_WAY_ENTRY

    %% ============================================================
    %% 7-WAY FAN-OUT (all sub-paths converge here)
    %% ============================================================
    SIX_WAY_ENTRY["⚡ 7-Way Fan-Out Entry<br/><i>All sub-paths converge</i>"]:::fanOutNode

    SIX_WAY_ENTRY --> SEVEN_FAN{7-Way Parallel<br/>Dispatcher}:::fanOutNode

    %% ── Branch 1: Chunking → Dense Embedding → 2-way storage ──
    SEVEN_FAN -->|"Branch 1"| CHUNK
    subgraph BRANCH_1["Branch 1: Chunking + Dense Vectors"]
        CHUNK["Chunking<br/><i>Split MD into chunks</i>"]:::parallelNode
        CHUNK --> DENSE_EMB["Dense Embedding<br/><i>Cohere / Jina / Qwen</i>"]:::parallelNode
        DENSE_EMB --> DENSE_FAN{Dense 2-Way<br/>Fan-Out}:::fanOutNode
        DENSE_FAN -->|"Neo4j"| CREATE_CHUNK["Create Chunk Nodes<br/><i>Neo4j storage</i>"]:::storageNode
        DENSE_FAN -->|"Milvus"| STORE_DENSE["Store Dense Vectors<br/><i>Milvus storage</i>"]:::storageNode
    end

    %% ── Branch 2: Sparse Embedding ──
    SEVEN_FAN -->|"Branch 2"| SPARSE_EMB
    subgraph BRANCH_2["Branch 2: Sparse Vectors"]
        SPARSE_EMB["Sparse Embedding<br/><i>BM25 Frozen</i>"]:::parallelNode
        SPARSE_EMB --> STORE_SPARSE["Store Sparse Vectors<br/><i>Milvus storage</i>"]:::storageNode
    end

    %% ── Branch 3: Auto Tagging ──
    SEVEN_FAN -->|"Branch 3"| AUTO_TAG
    subgraph BRANCH_3["Branch 3: Auto Tagging"]
        AUTO_TAG["Auto Tagging<br/><i>Claude Classifier</i>"]:::parallelNode
        AUTO_TAG --> WHITELIST["Whitelist Validator<br/><i>Filter by vocabulary</i>"]:::processNode
        WHITELIST --> STORE_TAGS["Store Tags<br/><i>Neo4j storage</i>"]:::storageNode
    end

    %% ── Branch 4: Store Fulltext MD ──
    SEVEN_FAN -->|"Branch 4"| STORE_MD
    subgraph BRANCH_4["Branch 4: Fulltext Storage"]
        STORE_MD["Store Fulltext MD<br/><i>Neo4j storage</i>"]:::storageNode
    end

    %% ── Branch 5: Quality Evaluation ──
    SEVEN_FAN -->|"Branch 5"| EVAL_GATE
    subgraph BRANCH_5["Branch 5: Quality Evaluation"]
        EVAL_GATE["Evaluation Gate<br/><i>Check if eval needed</i>"]:::evalNode
        EVAL_GATE --> EVAL_ROUTE{Evaluation<br/>Router}:::routingNode
        EVAL_ROUTE -->|"evaluate"| QUALITY_EVAL["Quality Evaluation<br/><i>LLM Document Evaluator</i>"]:::evalNode
        EVAL_ROUTE -->|"skip"| MOVE_CURATED_SKIP["Move to Curated<br/><i>MinIO: Staging → Curated</i>"]:::minioNode
        QUALITY_EVAL --> THRESHOLD{Threshold<br/>Router}:::routingNode
        THRESHOLD -->|"pass ≥ 0.7"| MOVE_CURATED["Move to Curated<br/><i>MinIO: Staging → Curated</i>"]:::minioNode
        THRESHOLD -->|"fail < 0.7"| MOVE_EVALUATED["Move to Evaluated<br/><i>MinIO: Staging → Evaluated</i>"]:::minioNode
    end

    %% ── Branch 6: Move Landing → Staging ──
    SEVEN_FAN -->|"Branch 6"| MOVE_L2S
    subgraph BRANCH_6["Branch 6: File Lifecycle"]
        MOVE_L2S["Move Landing → Staging<br/><i>MinIO bucket transition</i>"]:::minioNode
    end

    %% ── Branch 7: Notify Cleaning Service ──
    SEVEN_FAN -->|"Branch 7"| NOTIFY_CLEAN
    subgraph BRANCH_7["Branch 7: Notification"]
        NOTIFY_CLEAN["Notify Cleaning Service<br/><i>HTTP POST to cleaning svc</i>"]:::notifyNode
    end

    %% ============================================================
    %% COMPLETION
    %% ============================================================
    CREATE_CHUNK --> COMPLETE
    STORE_DENSE --> COMPLETE
    STORE_SPARSE --> COMPLETE
    STORE_TAGS --> COMPLETE
    STORE_MD --> COMPLETE
    MOVE_CURATED --> COMPLETE
    MOVE_CURATED_SKIP --> COMPLETE
    MOVE_EVALUATED --> COMPLETE
    MOVE_L2S --> COMPLETE
    NOTIFY_CLEAN --> COMPLETE

    COMPLETE([✅ Pipeline Complete<br/><i>Redis Pub/Sub notification</i>]):::endNode            