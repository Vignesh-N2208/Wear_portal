from fastapi import Request
from collections import Counter
import math

from urllib.parse import quote_plus
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body, FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os, re, uuid, asyncio, aiohttp, json
from typing import List, Dict, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, VectorizedQuery
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.storage.blob import BlobClient
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Removed incorrect import of 'session' from requests
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy import text
from sqlalchemy import select, func, or_
from sqlalchemy.orm import Bundle
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,  # class name in many SDK builds; alias of VectorSearchAlgorithmConfiguration
    VectorSearchProfile,
    LexicalAnalyzerName,
    CorsOptions,
)
from azure.search.documents.indexes import SearchIndexClient
import logging


from pydantic import BaseModel
from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions

# --- ChatAnswer endpoint with safety and retrieval ---
from azure.ai.contentsafety.models import AnalyzeTextOptions

# Environment variables for Content Safety
CS_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
CS_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY")

# Initialize Content Safety client (only if configured)
cs_client = None
if CS_ENDPOINT and CS_KEY:
    cs_client = ContentSafetyClient(CS_ENDPOINT, AzureKeyCredential(CS_KEY))


class ChatAnswerRequest(BaseModel):
    session_id: str
    user_msg: str


class ChatAnswerResponse(BaseModel):
    answer: str
    sources: List[Dict]


async def is_text_safe(user_text: str) -> bool:
    """Return False if text is unsafe (Medium/High severity), else True.
    If the service isn't configured or errors, default to True to avoid breaking UX."""
    if cs_client is None:
        return True
    try:
        options = AnalyzeTextOptions(text=user_text)
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: cs_client.analyze_text(options)
        )
        for category in getattr(result, "categories", []):
            sev = getattr(category, "severity", None)
            if sev in ("Medium", "High"):
                return False
        return True
    except Exception:
        return True


# -------------------------------------------------------------------
# Evaluation Endpoint for Model Metrics
# -------------------------------------------------------------------

logging.basicConfig(
    filename="evaluation_log.txt",
    level=logging.INFO,
    # Log format includes timestamp, severity, filename/line, and message
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("ingestion_pipeline")

# -------------------------------------------------------------------
# App & Env
# -------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    print("Application startup complete.")
    yield
    # Shutdown logic goes here
    print("Application shutdown complete.")


load_dotenv()
app = FastAPI(title="Wear Reports Chatbot")

# Add CORS middleware to allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["http://localhost:8080"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Azure Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
SEARCH_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")


# Helper: Ensure index exists, create if missing
def ensure_search_index(endpoint, index_name, key):
    index_client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    try:
        index_client.get_index(index_name)
    except Exception as e:
        if "ResourceNotFoundError" in str(type(e)) or "not found" in str(e).lower():
            # Define index schema (customize fields as needed)

            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-config",
                        kind="hnsw",
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vs-profile",
                        algorithm_configuration_name="hnsw-config",
                    )
                ],
            )

            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                    sortable=True,
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    analyzer_name=LexicalAnalyzerName.EN_LUCENE,
                ),
                SimpleField(
                    name="source",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="report_id",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="standard",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="implant_type",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="design_variant",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="material_pair",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True,
                ),
                SimpleField(
                    name="lab", type=SearchFieldDataType.String, filterable=True
                ),
                SimpleField(
                    name="date", type=SearchFieldDataType.String, filterable=True
                ),
                SimpleField(
                    name="total_wear_5mc",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
                SearchField(
                    name="contentVector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=3072,
                    vector_search_profile_name="vs-profile",
                ),
            ]
            cors = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
            index = SearchIndex(
                name=index_name,
                fields=fields,
                vector_search=vector_search,
                cors_options=cors,
            )
            index_client.create_index(index)
        else:
            raise
    return SearchClient(endpoint, index_name, AzureKeyCredential(key))


search_client = ensure_search_index(SEARCH_ENDPOINT, SEARCH_INDEX, SEARCH_KEY)

# Azure OpenAI
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
EMBED_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBED_DEPLOY = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
CHAT_ENDPOINT = os.getenv("AZURE_CHAT_ENDPOINT").rstrip("/")
CHAT_KEY = os.getenv("AZURE_CHAT_API_KEY")
CHAT_DEPLOY_HEAVY = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Azure Blob Storage
STORAGE_SAS = os.getenv("AZURE_STORAGE_SAS")
STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")

# Azure Document Intelligence
DI_EP = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
DI_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
di_client = DocumentIntelligenceClient(DI_EP, AzureKeyCredential(DI_KEY))


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------


MSSQL_HOST = os.getenv("MSSQL_HOST")
MSSQL_PORT = os.getenv("MSSQL_PORT", "1433")
MSSQL_USER = os.getenv("MSSQL_USER")
MSSQL_PASSWORD = os.getenv("MSSQL_PASSWORD")
MSSQL_DATABASE = os.getenv("MSSQL_DATABASE")

# SQLAlchemy async connection string for Azure SQL (aioodbc)

# Build a native ODBC connection string (Windows)
odbc_str = (
    "Driver={SQL Server};"
    f"Server=tcp:{MSSQL_HOST},{MSSQL_PORT};"
    f"Database={MSSQL_DATABASE};"
    f"Uid={MSSQL_USER};"
    f"Pwd={MSSQL_PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)
MSSQL_URL = f"mssql+aioodbc:///?odbc_connect={quote_plus(odbc_str)}"
engine = create_async_engine(MSSQL_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
Base = declarative_base()


class ChatTurn(BaseModel):
    session_id: str
    user_msg: str


class WearReport(Base):
    __tablename__ = "wear_reports"
    id = Column(Integer, primary_key=True, autoincrement=True)
    file = Column(String(256))
    report_id = Column(String(64))
    date = Column(String(32))
    standard = Column(String(64))
    implant_type = Column(String(128))
    design_variant = Column(String(128))
    material_pair = Column(String(128))
    machine = Column(String(128))
    freq = Column(String(64))
    temp = Column(String(64))
    lubricant = Column(String(128))
    duration = Column(String(64))
    total_wear_5mc = Column(String(64))
    wear_curve = Column(String)  # Store as NVARCHAR(MAX), stringified JSON


class ChatTurn(BaseModel):
    session_id: str
    user_msg: str


# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------


# Robust DDL logic: create table, add JSON constraint, create index

CREATE_TABLE_SQL = """
IF OBJECT_ID(N'[dbo].[wear_reports]', N'U') IS NULL
BEGIN
    CREATE TABLE [dbo].[wear_reports] (
        [id]               INT IDENTITY(1,1) PRIMARY KEY,
        [file]             NVARCHAR(256)     NULL,
        [report_id]        NVARCHAR(64)      NULL,
        [date]             NVARCHAR(32)      NULL,
        [standard]         NVARCHAR(64)      NULL,
        [implant_type]     NVARCHAR(128)     NULL,
        [design_variant]   NVARCHAR(128)     NULL,
        [material_pair]    NVARCHAR(128)     NULL,
        [machine]          NVARCHAR(128)     NULL,
        [freq]             NVARCHAR(64)      NULL,
        [temp]             NVARCHAR(64)      NULL,
        [lubricant]        NVARCHAR(128)     NULL,
        [duration]         NVARCHAR(64)      NULL,
        [total_wear_5mc]   NVARCHAR(64)      NULL,
        [wear_curve]       NVARCHAR(MAX)     NULL
    );
END
"""

ADD_JSON_CHECK_SQL = """
IF NOT EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = N'ck_wear_curve_is_json'
      AND parent_object_id = OBJECT_ID(N'[dbo].[wear_reports]')
)
BEGIN
    ALTER TABLE [dbo].[wear_reports]
        ADD CONSTRAINT [ck_wear_curve_is_json]
        CHECK ([wear_curve] IS NULL OR ISJSON([wear_curve]) = 1);
END
"""

ADD_INDEX_SQL = """
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = N'ix_wear_reports_report_id'
      AND object_id = OBJECT_ID(N'[dbo].[wear_reports]')
)
BEGIN
    CREATE INDEX [ix_wear_reports_report_id]
        ON [dbo].wear_reports ([report_id]);
END
"""


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.execute(text(CREATE_TABLE_SQL))
    async with engine.begin() as conn:
        await conn.execute(text(ADD_JSON_CHECK_SQL))
    async with engine.begin() as conn:
        await conn.execute(text(ADD_INDEX_SQL))


async def ingest_record_mssql(record: dict) -> int:
    # Ensure the table exists
    create_table_sql = """
    IF OBJECT_ID(N'[dbo].[wear_reports]', N'U') IS NULL
    BEGIN
        CREATE TABLE [dbo].[wear_reports] (
            [id]               INT IDENTITY(1,1) PRIMARY KEY,
            [file]             NVARCHAR(256)     NULL,
            [report_id]        NVARCHAR(64)      NULL,
            [date]             NVARCHAR(32)      NULL,
            [standard]         NVARCHAR(64)      NULL,
            [implant_type]     NVARCHAR(128)     NULL,
            [design_variant]   NVARCHAR(128)     NULL,
            [material_pair]    NVARCHAR(128)     NULL,
            [machine]          NVARCHAR(128)     NULL,
            [freq]             NVARCHAR(64)      NULL,
            [temp]             NVARCHAR(64)      NULL,
            [lubricant]        NVARCHAR(128)     NULL,
            [duration]         NVARCHAR(64)      NULL,
            [total_wear_5mc]   NVARCHAR(64)      NULL,
            [wear_curve]       NVARCHAR(MAX)     NULL
        );
    END
    """

    async with AsyncSessionLocal() as session:
        # Ensure the table exists
        await session.execute(text(create_table_sql))
        await session.commit()

        # Prepare parameters and ensure wear_curve is stringified
        params = dict(record)
        if "wear_curve" in params and params["wear_curve"] is not None:
            if not isinstance(params["wear_curve"], str):
                try:
                    params["wear_curve"] = json.dumps(params["wear_curve"])
                except Exception:
                    params["wear_curve"] = str(params["wear_curve"])

        insert_sql = """
        INSERT INTO [dbo].[wear_reports] (
            [file], [report_id], [date], [standard], [implant_type], [design_variant],
            [material_pair], [machine], [freq], [temp], [lubricant], [duration],
            [total_wear_5mc], [wear_curve]
        ) OUTPUT inserted.id VALUES (
            :file, :report_id, :date, :standard, :implant_type, :design_variant,
            :material_pair, :machine, :freq, :temp, :lubricant, :duration,
            :total_wear_5mc, :wear_curve
        );
        """
        result = await session.execute(text(insert_sql), params)
        row = result.fetchone()
        await session.commit()
        # row[0] is the inserted id
        return row[0] if row else None


@app.post("/db/ingest_record")
async def db_ingest_record(record: dict):
    obj_id = await ingest_record_mssql(record)
    return {"id": obj_id, "message": "Record ingested to Azure SQL"}


# Endpoint: Get all records for dashboard
@app.get("/db/records")
async def db_get_records():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            WearReport.__table__.select().order_by(WearReport.id.desc())
        )
        rows = result.fetchall()
        # Convert SQLAlchemy Row objects to dicts
        records = [dict(row._mapping) for row in rows]
    return {"records": records}


async def get_db_data_for_comparison(user_msg: str) -> Optional[List[Dict]]:
    """
    Searches the SQL database for records relevant to comparison/statistical queries.
    This bypasses the search index for structured data retrieval.
    """
    msg = user_msg.lower()

    # Simple keyword filtering for common comparison fields
    # This logic can be expanded using a dedicated NLP tool if needed
    filters = []

    if "hip" in msg and "knee" not in msg and "shoulder" not in msg:
        filters.append(WearReport.implant_type.ilike("%hip%"))
    if "knee" in msg and "hip" not in msg and "shoulder" not in msg:
        filters.append(WearReport.implant_type.ilike("%knee%"))

    # Material pairing filter
    if "xlpe" in msg or "cross-linked" in msg:
        filters.append(
            or_(
                WearReport.material_pair.ilike("%XLPE%"),
                WearReport.material_pair.ilike("%cross-linked%"),
            )
        )
    elif "uhmwpe" in msg:
        filters.append(WearReport.material_pair.ilike("%UHMWPE%"))
    elif "ceramic" in msg:
        filters.append(WearReport.material_pair.ilike("%Ceramic%"))
    elif "metal" in msg or "cocrmo" in msg:
        filters.append(
            or_(
                WearReport.material_pair.ilike("%Metal%"),
                WearReport.material_pair.ilike("%CoCrMo%"),
            )
        )

    if not filters:
        # If no specific filters found, return None to proceed with RAG search
        return None

    async with AsyncSessionLocal() as db_session:
        # Select all relevant columns, focusing on data points
        # Using Bundle for clarity, though not strictly necessary here
        report_data = Bundle(
            "report_data",
            WearReport.file,
            WearReport.report_id,
            WearReport.implant_type,
            WearReport.material_pair,
            WearReport.total_wear_5mc,
        )

        # Apply filters if they exist
        stmt = select(report_data).limit(
            50
        )  # Limit to a reasonable number for comparison
        if filters:
            stmt = select(report_data).where(or_(*filters)).limit(50)

        result = await db_session.execute(stmt)
        rows = result.fetchall()

        # Convert SQLAlchemy Row objects to dicts
        records = []
        for row in rows:
            record_dict = row._asdict()["report_data"]._mapping
            # Only include records with a quantifiable wear rate
            if record_dict["total_wear_5mc"]:
                try:
                    float(record_dict["total_wear_5mc"])
                    records.append(record_dict)
                except (TypeError, ValueError):
                    continue

        return records if records else None


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
async def aoai_embeddings(
    session: aiohttp.ClientSession, texts: List[str]
) -> List[List[float]]:
    url = f"{EMBED_ENDPOINT}/openai/deployments/{EMBED_DEPLOY}/embeddings?api-version={API_VERSION}"
    async with session.post(
        url,
        headers={"api-key": EMBED_KEY, "Content-Type": "application/json"},
        json={"input": texts},
    ) as resp:
        if resp.status >= 400:
            raise HTTPException(
                status_code=resp.status,
                detail=f"Embeddings failed: {await resp.text()}",
            )
        data = await resp.json()
        return [item["embedding"] for item in data["data"]]


async def aoai_embed_one(session: aiohttp.ClientSession, text: str) -> List[float]:
    return (await aoai_embeddings(session, [text]))[0]


async def aoai_chat_stream(
    session: Optional[aiohttp.ClientSession],
    system_prompt: str,
    user_message: str,
    context: str,
):
    """
    Streams a response from Azure OpenAI Chat Completion with RAG context.
    The function handles creating and closing its own aiohttp session if none is provided,
    preventing the 'Session is closed' error during streaming.
    """

    # 1. Determine Session Strategy
    local_session = False
    if session is None:
        session = aiohttp.ClientSession()
        local_session = True

    try:
        url = (
            f"{CHAT_ENDPOINT}/openai/deployments/{CHAT_DEPLOY_HEAVY}/chat/completions"
            f"?api-version={API_VERSION}"
        )

        # Construct the full prompt for the LLM
        full_user_message = (
            f"User Query: {user_message}\n\n"
            f"Context from Index:\n---\n{context}\n---"
            "Based on the CONTEXT provided above, answer the USER QUERY. "
            "If the context does not contain the answer, state that you cannot find the relevant information."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_message},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": True,
        }

        # 2. Make the Streaming POST Request using the (potentially new) session
        async with session.post(
            url,
            headers={"api-key": CHAT_KEY, "Content-Type": "application/json"},
            json=payload,
        ) as resp:
            if resp.status >= 400:
                # Read the full response text for the error detail
                error_text = await resp.text()

                # Check for rate limiting or other common issues
                if resp.status == 429:
                    error_detail = (
                        f"Chat API failed: Rate limit exceeded. Details: {error_text}"
                    )
                else:
                    error_detail = (
                        f"Chat API failed (Status {resp.status}): {error_text}"
                    )

                # Yield the error to the client instead of raising an HTTPException
                # because we are in a generator that is already executing the response.
                yield f"\n[API ERROR: {error_detail}]\n"
                return  # Stop the generator

            # 3. Generator function to parse and yield chunks
            async for chunk in resp.content.iter_any():
                if chunk:
                    try:
                        # Azure OpenAI uses Server-Sent Events (SSE) format
                        lines = chunk.decode("utf-8").splitlines()
                        for line in lines:
                            if not line.strip():
                                continue  # Skip empty lines
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    break
                                # Only try to parse if it looks like JSON
                                if data.startswith("{") and data.endswith("}"):
                                    try:
                                        parsed = json.loads(data)
                                        choices = parsed.get("choices", [])
                                        if choices and "delta" in choices[0]:
                                            content = choices[0]["delta"].get("content")
                                            if content:
                                                yield content
                                    except Exception:
                                        continue
                    except Exception as e:
                        # Log error but keep streaming
                        print(f"Error parsing SSE chunk: {e}")
                        continue

    except Exception as e:
        # Catch connection errors, DNS lookup failures, etc.
        yield f"\n[STREAMING ERROR: An unexpected error occurred: {type(e).__name__} - {str(e)}]\n"

    finally:
        # 4. Close the session ONLY if we created it locally
        if local_session:
            await session.close()


async def retrieve_context(
    query: str, session: aiohttp.ClientSession, k: int = 8
) -> List[Dict]:
    qvec = await aoai_embed_one(session, query)
    vq = VectorizedQuery(vector=qvec, k_nearest_neighbors=k, fields="contentVector")
    results = search_client.search(
        search_text=query,
        vector_queries=[vq],
        query_type=QueryType.SIMPLE,
        select=[
            "id",
            "content",
            "source",
            "report_id",
            "standard",
            "implant_type",
            "design_variant",
            "material_pair",
        ],
        top=k,
    )
    docs = []
    for r in results:
        docs.append(r)
    return docs


def parse_text_to_record(text: str, file_name: str):
    pat = {
        "report_id": r"Report ID:\s*([A-Z0-9\-]+)",
        "date": r"Date:\s*([0-9\-]+)",
        "standard": r"(ISO\s+[0-9\-]+)",
        "implant_type": r"Implant Type\s*([^\n]+)",
        "design_variant": r"Design Variant\s*([^\n]+)",
        "material_pair": r"Material Pairing\s*([^\n]+)",
        "machine": r"Machine Used:\s*([^\n]+)",
        "freq": r"Testing Frequency:\s*([^\n]+)",
        "temp": r"Temperature:\s*([^\n]+)",
        "lubricant": r"Lubricant:\s*([^\n]+)",
        "duration": r"Testing Duration:\s*([^\n]+)",
        "total_wear_5mc": r"Total Wear\s*\(5\s*Mc\)\s*([0-9\.]+)\s*mm³",
    }
    rec = {"file": file_name}
    for k, p in pat.items():
        m = re.search(p, text)
        rec[k] = m.group(1).strip() if m else None
    # wear curve lines "0.5 Mc 1.7366"
    curve = []
    for mc, val in re.findall(r"([0-9\.]+)\s*Mc\s*([0-9\.]+)", text):
        if mc != "." and val != ".":
            try:
                curve.append({"mc": float(mc), "cum_mm3": float(val)})
            except:
                pass
    curve.sort(key=lambda x: x["mc"])
    rec["wear_curve"] = curve
    # Set total_wear_5mc from wear_curve (mc == 5.0), fallback to last value
    mc_5_val = next((str(c["cum_mm3"]) for c in curve if c["mc"] == 5.0), None)
    if mc_5_val:
        rec["total_wear_5mc"] = mc_5_val
    elif curve:
        rec["total_wear_5mc"] = str(curve[-1]["cum_mm3"])

    return rec


def chunk_record(rec: Dict) -> List[Dict]:
    base = f"""Report ID: {rec.get('report_id')}
Standard: {rec.get('standard')}
Implant Type: {rec.get('implant_type')}
Design: {rec.get('design_variant')}
Materials: {rec.get('material_pair')}
Machine: {rec.get('machine')}
Frequency: {rec.get('freq')}
Temperature: {rec.get('temp')}
Lubricant: {rec.get('lubricant')}
Duration: {rec.get('duration')}"""
    chunks = [
        {
            "text": base.strip(),
            "source": rec.get("file"),
            "report_id": rec.get("report_id"),
            "standard": rec.get("standard"),
            "implant_type": rec.get("implant_type"),
            "design_variant": rec.get("design_variant"),
            "material_pair": rec.get("material_pair"),
        }
    ]
    wc = rec.get("wear_curve") or []
    if wc:
        curve_txt = "Wear curve: " + "; ".join(
            [f"{p['mc']}Mc={p['cum_mm3']}mm3" for p in wc]
        )
        chunks.append(
            {
                "text": curve_txt,
                "source": rec.get("file"),
                "report_id": rec.get("report_id"),
                "standard": rec.get("standard"),
                "implant_type": rec.get("implant_type"),
                "design_variant": rec.get("design_variant"),
                "material_pair": rec.get("material_pair"),
            }
        )
    return chunks


TABLE_CUES = {"list", "compare", "table", "show", "export", "sort", "rank", "trend"}
SINGLE_CUES = {
    "what happened on",
    "what happened in",
    "show report",
    "show me report",
    "test ",
    "wt-",
    "iso-",
    "report ",
    "retrieve",
    "find the report for",
}
TEST_ID_PAT = re.compile(
    r"(?:wt-\d{4}-\d{3}|iso-\d{4}-\d{4}|test\d{3,})", re.IGNORECASE
)


def infer_intent(user_msg: str) -> dict:
    """Detect whether the user wants a list/compare/table vs a single-record narrative."""
    msg = user_msg.strip().lower()
    wants_table = any(cue in msg for cue in TABLE_CUES)
    single_lookup = any(cue in msg for cue in SINGLE_CUES) or bool(
        TEST_ID_PAT.search(msg)
    )
    is_comparison = "compare" in msg
    is_statistical = any(
        k in msg for k in ["average", "median", "distribution", "standard deviation"]
    )
    is_trend = "trend" in msg or "evolution" in msg
    return {
        "wants_table": wants_table,
        "single_lookup": single_lookup,
        "is_comparison": is_comparison,
        "is_statistical": is_statistical,
        "is_trend": is_trend,
    }


def should_use_table(intent: dict, docs: list, user_msg: str) -> bool:
    """Return True if table output is explicitly helpful; otherwise prefer concise narrative."""
    if intent["single_lookup"]:
        return False
    if intent["wants_table"]:
        return True
    if intent["is_comparison"] or intent["is_trend"]:
        return True
    multi_phrasing = any(
        p in user_msg.lower() for p in ["show all", "list all", "find all"]
    )
    return multi_phrasing and len(docs) > 1


def build_narrative(d: dict) -> str:
    """Compose a brief single-record summary using only fields present in the index."""
    parts = []
    if d.get("source"):
        parts.append(f"Source: {d['source']}")
    if d.get("implant_type"):
        parts.append(f"Implant: {d['implant_type']}")
    if d.get("date"):
        parts.append(f"Date: {d['date']}")
    if d.get("standard"):
        parts.append(f"Standard: {d['standard']}")
    if d.get("material_pair"):
        parts.append(f"Materials: {d['material_pair']}")

    # Optional extras — only if present in the record
    for extra in [
        "load",
        "frequency",
        "cycles",
        "simulator",
        "validation_status",
        "lab",
        "equipment_id",
    ]:
        if d.get(extra):
            label = extra.replace("_", " ").title()
            parts.append(f"{label}: {d[extra]}")

    return "; ".join(parts) if parts else "No indexed fields available for this test."


def build_table(docs: list) -> str:
    """Minimal markdown table for multi-record comparison."""
    header = (
        "| Source | Implant Type | Date | Standard | Material Pair |\n"
        "|--------|--------------|------|----------|---------------|"
    )
    rows = []
    for d in docs:
        rows.append(
            f"| {d.get('source','')} | {d.get('implant_type','')} | {d.get('date','')} "
            f"| {d.get('standard','')} | {d.get('material_pair','')} | {d.get('total_wear_5mc','')} |"
        )
    return "\n".join([header, *rows])


# -------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------


@app.post("/ingest/preview")
async def ingest_preview(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed.")
    data = await file.read()
    poller = di_client.begin_analyze_document(model_id="prebuilt-layout", body=data)
    result = poller.result()
    full_text = "\n".join(
        [line.content for page in result.pages for line in page.lines]
    )
    rec = parse_text_to_record(full_text, file.filename)
    print(f"Parsed record: {rec}")
    return JSONResponse({"fields": rec})


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile, file_name: Optional[str] = Form(None)):
    if not file.filename.lower().endswith(".pdf"):
        # Log: Invalid File Type
        logger.error(
            f"File: {file.filename} | Status: Failure | Reason: Invalid file type."
        )
        raise HTTPException(status_code=400, detail="Only PDF files allowed.")

    data = await file.read()
    target_name = file_name or file.filename

    # --- 1. Document Intelligence Extraction & Parsing ---
    try:
        poller = di_client.begin_analyze_document(model_id="prebuilt-layout", body=data)
        result = poller.result()
        full_text = "\n".join(
            [line.content for page in result.pages for line in page.lines]
        )
        rec = parse_text_to_record(full_text, target_name)

        # Data Quality Validation Log: Check for missing critical fields
        if not rec.get("report_id") or not rec.get("total_wear_5mc"):
            logger.warning(
                f"File: {target_name} | Status: Warning | Reason: Missing critical fields (ReportID: {rec.get('report_id')}, WearRate: {rec.get('total_wear_5mc')})."
            )

    except Exception as e:
        # Log: Extraction Failure
        logger.error(
            f"File: {target_name} | Status: Failure | Reason: DI Extraction failed. Error: {e}"
        )
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

    chunks = chunk_record(rec)

    # --- 2. Check for duplicate by report_id in DB ---
    async with AsyncSessionLocal() as session:
        existing = None
        if rec.get("report_id"):
            q = await session.execute(
                WearReport.__table__.select().where(
                    WearReport.report_id == rec["report_id"]
                )
            )
            existing = q.fetchone()
        if existing:
            # Log: DB Duplicate Found (Skipped)
            logger.info(
                f"File: {target_name} | Status: Skipped | Reason: DB duplicate found for Report ID '{rec['report_id']}'."
            )
            return JSONResponse(
                {
                    "detail": f"Record with report_id '{rec['report_id']}' already exists in DB. Upload skipped."
                },
                status_code=409,
            )

    # --- 3. Check for duplicate in blob storage ---
    bc = BlobClient.from_connection_string(
        STORAGE_SAS, container_name=STORAGE_CONTAINER, blob_name=target_name
    )
    try:
        if bc.exists():
            # Log: Blob Duplicate Found (Skipped)
            logger.info(
                f"File: {target_name} | Status: Skipped | Reason: Blob duplicate found."
            )
            return JSONResponse(
                {
                    "detail": f"File '{target_name}' already exists in blob storage. Upload skipped."
                },
                status_code=409,
            )
    except Exception as e:
        # Log: Unexpected Blob Check Error (will proceed to upload attempt)
        logger.warning(
            f"File: {target_name} | Status: Warning | Reason: Blob check error: {e}"
        )
        pass

    # --- 4. Upload to blob storage ---
    async def _upload_blob():
        bc.upload_blob(data, overwrite=True)

    try:
        await _upload_blob()
    except Exception as e:
        # Log: Blob Upload Failure
        logger.error(
            f"File: {target_name} | Status: Failure | Reason: Blob upload failed. Error: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Blob upload failed: {e}")

    # --- 5. Ingest parsed record into Azure SQL ---
    try:
        await ingest_record_mssql(rec)
    except Exception as e:
        # Log: SQL Ingestion Failure
        logger.error(
            f"File: {target_name} | Status: Failure | Reason: SQL Ingestion failed. Error: {e}"
        )
        # At this point, you may want compensating logic (e.g., delete the blob).
        raise HTTPException(status_code=500, detail=f"SQL ingestion failed: {e}")

    # --- 6. Indexing (Embeddings & Azure Search) ---
    try:
        async with aiohttp.ClientSession() as session:
            docs = [
                {
                    "id": str(uuid.uuid4()),
                    "content": c["text"],
                    "contentVector": v,
                    "source": c.get("source"),
                    "report_id": c.get("report_id"),
                    "standard": c.get("standard"),
                    "implant_type": c.get("implant_type"),
                    "design_variant": c.get("design_variant"),
                    "material_pair": c.get("material_pair"),
                }
                for c, v in zip(
                    chunks, await aoai_embeddings(session, [c["text"] for c in chunks])
                )
            ]
            search_client.upload_documents(docs)

        # --- FINAL SUCCESS LOG ---
        logger.info(
            f"File: {target_name} | Status: Success | Report ID: {rec.get('report_id')}. All steps complete."
        )

    except Exception as e:
        # Log: Azure Search Indexing Failure (Partial success—DB has data, Index doesn't)
        logger.error(
            f"File: {target_name} | Status: Failure | Reason: Azure Search indexing failed. Error: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Azure Search indexing failed: {e}"
        )

    return JSONResponse({"message": f"File {target_name} ingested and indexed."})


@app.post("/chat/answer", response_model=ChatAnswerResponse)
async def chat_answer(turn: ChatAnswerRequest):
    # 0) Safety & off-topic guardrails
    is_safe = await is_text_safe(turn.user_msg)
    if not is_safe:
        return ChatAnswerResponse(
            answer=(
                "I can't process that request. Please ask about wear test data—"
                "for example: hip/knee implants, XLPE/UHMWPE, ISO standards, or a report ID like WT-2024-001."
            ),
            sources=[],
        )

    # 1) Intent detection and retrieval (reuse existing helpers)
    intent = infer_intent(turn.user_msg)
    db_records = None
    rag_docs = []
    async with aiohttp.ClientSession() as temp_session:
        if intent["is_statistical"] or intent["is_comparison"] or intent["wants_table"]:
            db_records = await get_db_data_for_comparison(turn.user_msg)
        if not db_records:
            rag_docs = await retrieve_context(turn.user_msg, temp_session, k=8)

    # 2) Build context string + collect sources
    if db_records:
        context_header = (
            f"Structured Data from Azure SQL ({len(db_records)} records found):"
        )
        db_lines = [
            (
                f"File: {d.get('file','N/A')}; Report ID: {d.get('report_id','N/A')}; "
                f"Implant: {d.get('implant_type','N/A')}; Materials: {d.get('material_pair','N/A')}; "
                f"Total Wear (5Mc): {d.get('total_wear_5mc','N/A')} mm³"
            )
            for d in db_records
        ]
        rag_context = f"{context_header}\n---\n" + "\n".join(db_lines)
        sources = [
            {"source": d.get("file"), "report_id": d.get("report_id")}
            for d in db_records
        ]
    elif rag_docs:
        context_header = f"Vector Search Context ({len(rag_docs)} chunks found):"
        chunks = [
            (
                f"[Source: {d.get('source','N/A')}, Report ID: {d.get('report_id','N/A')}] "
                f"Content: {d.get('content','')}"
            )
            for d in rag_docs
        ]
        rag_context = f"{context_header}\n---\n" + "\n---\n".join(chunks)
        sources = [
            {"source": d.get("source"), "report_id": d.get("report_id")}
            for d in rag_docs
        ]
    else:
        return ChatAnswerResponse(
            answer=(
                "I couldn't find matching test data. Please refine with specific materials, implant types, "
                "or a report ID (e.g., 'hip XLPE 2023' or 'WT-2024-001')."
            ),
            sources=[],
        )

    # 3) Non-stream Azure OpenAI chat completion
    url = (
        f"{CHAT_ENDPOINT}/openai/deployments/{CHAT_DEPLOY_HEAVY}/chat/completions"
        f"?api-version={API_VERSION}"
    )
    system_prompt = (
        "You are an expert assistant for orthopedic wear testing data from DePuy Synthes. "
        "Answer strictly based on the provided CONTEXT. If the answer isn't in context, "
        "say: 'The required information is not available in the current data.' Use concise Markdown; "
        "include comparison tables when appropriate; always avoid hallucination."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User Query: {turn.user_msg}\n\nContext:\n---\n{rag_context}\n---",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers={"api-key": CHAT_KEY, "Content-Type": "application/json"},
            json=payload,
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                return ChatAnswerResponse(
                    answer=f"Chat API error ({resp.status}): {text}", sources=[]
                )
            data = await resp.json()
            answer_text = data["choices"][0]["message"]["content"].strip()

    return ChatAnswerResponse(answer=answer_text, sources=sources)


@app.post("/chat/stream")
async def chat(turn: ChatTurn = Body(...)):

    # 1) Log user queries (existing logic)
    try:
        with open("user_query_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"{turn.session_id}\t{turn.user_msg.strip()}\n")
    except Exception:
        pass

    user_msg_raw = turn.user_msg.strip()
    user_msg = user_msg_raw.lower()

    # 2) System prompt (Enhanced for RAG and table/comparison tasks)
    system_prompt = (
        "You are an expert assistant for orthopedic wear testing data from DePuy Synthes. "
        "Your primary goal is to provide concise, accurate, and scientifically sound answers based *only* on the provided 'CONTEXT'. "
        "If the context contains multiple records, you MUST synthesize the results, perform calculations (like averages or comparisons), "
        "and present the data in a clear, formatted **Markdown table or list**. "
        "Always state the source file for the data. If the information is not in the context, state clearly: 'The required information is not available in the current data.' "
        "Strictly avoid hallucination. The data concerns wear testing for orthopaedic implants."
    )

    # 3) Intent Detection & Query Building
    intent = infer_intent(user_msg_raw)

    # Detect specific test IDs (e.g., WT-2024-001) for search query prioritization
    test_id_match = TEST_ID_PAT.search(user_msg)
    search_query = test_id_match.group(0) if test_id_match else user_msg_raw

    # --- Data Source Decision & Retrieval ---
    db_records = None
    rag_docs = []
    rag_context = ""
    async with aiohttp.ClientSession() as temp_session:
        if intent["is_statistical"] or intent["is_comparison"] or intent["wants_table"]:
            # Prioritize database retrieval for accurate comparison/statistical queries
            db_records = await get_db_data_for_comparison(user_msg_raw)

        if not db_records:
            # Fallback to vector search (RAG) for narrative questions or when DB filters fail
            rag_docs = await retrieve_context(search_query, temp_session, k=8)

    # 4) Context Building for the LLM
    if db_records:
        # A) Build context from structured DB records
        context_header = (
            f"Structured Data from Azure SQL ({len(db_records)} records found):"
        )

        # Format DB records into a context string (e.g., CSV-like or detailed list)
        db_context_list = []
        for d in db_records:
            db_context_list.append(
                f"File: {d.get('file', 'N/A')}; Report ID: {d.get('report_id', 'N/A')}; "
                f"Implant: {d.get('implant_type', 'N/A')}; Materials: {d.get('material_pair', 'N/A')}; "
                f"Total Wear (5Mc): {d.get('total_wear_5mc', 'N/A')} mm³"
            )
        rag_context = f"{context_header}\n---\n" + "\n".join(db_context_list)

    elif rag_docs:
        # B) Build context from indexed RAG documents
        context_header = f"Vector Search Context ({len(rag_docs)} chunks found):"
        # Concatenate content and metadata for the LLM
        context_chunks = []
        for d in rag_docs:
            context_chunks.append(
                f"[Source: {d.get('source', 'N/A')}, Report ID: {d.get('report_id', 'N/A')}] "
                f"Content: {d.get('content', '')}"
            )
        rag_context = f"{context_header}\n---\n" + "\n---\n".join(context_chunks)

    # 5) Streaming Response
    if rag_context:
        # Use LLM to synthesize the final answer based on the constructed context
        async def stream_with_session_and_log():
            response_chunks = []
            async with aiohttp.ClientSession() as session:
                async for chunk in aoai_chat_stream(
                    session, system_prompt, user_msg_raw, rag_context
                ):
                    response_chunks.append(chunk)
                    yield chunk
            response_text = "".join(response_chunks).strip()
            # Log to user_query_log.txt as before
            try:
                with open("user_query_log.txt", "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"{turn.session_id}\t{turn.user_msg.strip()}\t{response_text}\n"
                    )
            except Exception:
                pass

        return StreamingResponse(stream_with_session_and_log(), media_type="text/plain")

    # 6) Fallback: No data found
    async def fallback_streamer():
        fallback_msg = "I couldn't find any matching test data in the system. Could you refine your query with specific materials, implant types, or report IDs (e.g., 'hip 2023 XLPE' or 'WT-2024-001')?"
        for char in fallback_msg:
            yield char
            await asyncio.sleep(0.001)

    return StreamingResponse(fallback_streamer(), media_type="text/plain")


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "search_index": SEARCH_INDEX,
        "embed_deploy": EMBED_DEPLOY,
        "chat_heavy": CHAT_DEPLOY_HEAVY,
    }
