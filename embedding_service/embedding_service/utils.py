import os
import json
import asyncio
from openai import OpenAI
from docling.document_converter import DocumentConverter 
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_core.documents import Document
# from .summary import run_summary_flow
from .db import db
from .summary_agent import agent
from .redis_client import redis_conn
from .database import get_db_session


def document_loader(source: str):
    try:
        converter = DocumentConverter()
        loader = DoclingLoader(file_path=source,converter=converter)
        docs= loader.load()
        chunks = []
        for ind,doc in enumerate(docs):
            doc.metadata.update({'chunk_index': ind})
            chunks += [{"metadata": json.dumps(doc.metadata),
                        "text": doc.page_content}]            
        return chunks
    except Exception as e:
        print(f"[!] Error loading document {source}: {e}")
        return []

def event_loader(event):
    chunks = []
    title = event['title']
    description = event['description']
    start_datetime = event['start_datetime']
    end_datetime = event['end_datetime']
    tag_note = event['tag_note']

    text = f"""Event Title: {title}
    Overview: {description}
    Start Date and time : {start_datetime}
    End Date and time : {end_datetime}

    TAG Note : {tag_note}

    """
    chunks.append(
        {
        'text': text,
        'metadata':json.dumps({'title':title}),
        'event_id': event['event_id']
        }
    )
    return chunks

def stakeholder_loader(stakeholders):
    chunks=[]
    name = stakeholders.get('name')
    org_data = stakeholders.get('org', {})
    org = org_data.get('org')
    title = org_data.get('title')
    bio = stakeholders.get('bio')

    metadata = org_data.copy()
    metadata['name'] = name

    text = f"""Stakeholder Title: {title}
    Stakeholder Name: {name}
    Organization: {org}
    Stakeholder Biography: {bio}"""
    chunks.append(
        {
        'text': text,
        'metadata':json.dumps(metadata),
        'stakeholder_id': stakeholders['stakeholder_id']
        }
    )
    return chunks

def analysis_loader(analysis):
    chunks=[]
    heading = analysis.get('heading')
    body = analysis.get('body')
    summary = analysis.get('summary')

    # headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
    headers_to_split_on = []
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size = 600,
        chunk_overlap = 200,
        elements_to_preserve=["table","ul","ol"],)
    documents = splitter.split_text(body)
    for ind,doc in enumerate(documents):
        # print(doc)
        text = doc.page_content
        metadata = {'heading': heading,'chunk_index': ind}
        chunks.append({'text': text,'metadata':json.dumps(metadata),'analysis_id': analysis['analysis_id']})
    if not chunks:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200, is_separator_regex = ["\n\n"])
        texts = text_splitter.split_text(body)
        for ind,text in enumerate(texts):
            metadata = {'heading': heading,'chunk_index': ind}
            chunks.append({'text': text,'metadata':json.dumps(metadata),'analysis_id': analysis['analysis_id']})
    # if summary:
    #     text = f"""{heading}
    #     Summary: {summary}"""
    #     chunks.append({'text': text,'metadata':json.dumps(metadata),'analysis_id': analysis['analysis_id']})   
    return chunks

def create_embedding(chunks):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv("EMBEDDING_API_BASE"),)
    models = client.models.list()
    model = models.data[0].id
    responses = client.embeddings.create(input=[chunk['text'] for chunk in chunks], model=model)
    embeddings = [response.embedding for response in responses.data]
    return embeddings

async def process_document_embed(session, documents):
    print("Document IDs:", documents)
    documents = await db.get_embeddings_document(session,documents)
    processed_document = {}
    for document in documents:
        chunks_data = document_loader(document['path'])
        if not chunks_data:
            processed_document[document['document_id']] = False
            continue
        # status_flag = await process_summary_document(session, document['document_id'], chunks_data)
        # print(f"Document {document['document_id']} summary processed and saved: {status_flag}")
        status_flag = await process_document_embeddings(session, document['document_id'], chunks_data)
        print(f"Document {document['document_id']} embeddings processed and saved: {status_flag}")
        if status_flag:
            redis_conn.raw().rpush("summary_queue", document['document_id'])
        processed_document[document['document_id']] = status_flag
    return processed_document

# run backgroud task for summary flow  
# async def process_summary_document(session, doc_id, chunks_data):
#     if not chunks_data:
#         return False
#     print(f"length of chunks for document {doc_id}: {len(chunks_data)}")
#     documents = [Document(page_content=chunk['text'], metadata=json.loads(chunk['metadata'])) for chunk in chunks_data]
#     summary_task = await agent.ainvoke(input={"documents": documents})
#     final_summary = summary_task["final_summary"]
#     status_flag = await db.add_document_summaries(session, doc_id, final_summary)
#     return status_flag

async def process_summary_document(session, doc_id):
    chunks_data = await db.get_document_chunks(session, doc_id)
    if not chunks_data:
        return False
    print(f"length of chunks for document {doc_id}: {len(chunks_data)}")
    documents = [Document(page_content=chunk['text']) for chunk in chunks_data]
    summary_task = await agent.ainvoke(input={"documents": documents})
    final_summary = summary_task["final_summary"]
    status_flag = await db.add_document_summaries(session, doc_id, final_summary)
    return status_flag

async def process_document_embeddings(session, doc_id, chunks_data):
    if not chunks_data:
        return False
    embeddings = create_embedding(chunks_data)
    for chunk, embedding in zip(chunks_data, embeddings):
        chunk['embedding'] = embedding
        chunk['document_id'] = doc_id
    print(f"length of chunks for document {doc_id}: {len(chunks_data)}")
    status_flag = await db.add_document_embeddings(session, chunks_data)
    return status_flag

async def process_event_embed(session, events):
    print("Event IDs:", events)
    events = await db.get_embeddings_event(session, events)
    processed_event = {}
    chunks_data = []
    for event in events:
        event_id = event['event_id']
        print(f"Processing event ID: {event_id}")
        processed_event[event_id] = False
        chunks = event_loader(event)
        chunks_data.extend(chunks)
    if chunks_data:  
        embeddings = create_embedding(chunks_data)
        for chunk, embedding in zip(chunks_data, embeddings):
            chunk['embedding'] = embedding
        status_flag = await db.add_event_embeddings(session, chunks_data)
        for event_id in processed_event:        
            print(f"Event {event_id} processed and embedding saved: {status_flag}")
            processed_event[event_id] = status_flag
        
    return processed_event

async def process_stakeholder_embed(session, stakeholders):
    print("Stakeholder IDs:", stakeholders)
    stakeholders = await db.get_embeddings_stakeholder(session, stakeholders)
    processed_stakeholders={}
    chunks_data=[]
    for stakeholder in stakeholders:
        stakeholder_id = stakeholder['stakeholder_id']
        print('Processing Stakeholder ID: ', stakeholder_id)
        processed_stakeholders[stakeholder_id] = False 
        chunks=stakeholder_loader(stakeholder)
        chunks_data.extend(chunks)
    if chunks_data:
        embeddings = create_embedding(chunks_data)
        for chunk, embedding in zip(chunks_data, embeddings):
            chunk['embedding'] = embedding
        status_flag = await db.add_stakeholder_embeddings(session, chunks_data)
        for stakeholder_id in processed_stakeholders:
            print(f'Stakeholder {stakeholder_id} Processed and embedings saved {status_flag}')
            processed_stakeholders[stakeholder_id] = status_flag
    return processed_stakeholders

async def process_analysis_embed(session, analyses):
    print("analysis IDs:", analyses)
    analyses = await db.get_embeddings_analysis(session, analyses)
    processed_analyses={}
    chunks_data=[]
    for analysis in analyses:
        analysis_id = analysis['analysis_id']
        print('Processing analysis ID: ', analysis_id)
        processed_analyses[analysis_id] = False 
        chunks=analysis_loader(analysis)
        chunks_data.extend(chunks)

    if chunks_data:
        embeddings = create_embedding(chunks_data)
        for chunk, embedding in zip(chunks_data, embeddings):

            chunk['embedding'] = embedding
        status_flag = await db.add_analysis_embeddings(session, chunks_data)
        for analysis_id in processed_analyses:
            print(f'Analysis {analysis_id} Processed and embedings saved {status_flag}')
            processed_analyses[analysis_id] = status_flag
    return processed_analyses


async def process_embed(session, resources, resource_type):
    if resource_type == "Document":
        return await process_document_embed(session, resources)
    elif resource_type == "Calendar":
        return await process_event_embed(session, resources)
    elif resource_type == "Stakeholder":
        return await process_stakeholder_embed(session, resources)
    elif resource_type == "Analysis":
        return await process_analysis_embed(session, resources)
    else:
        raise ValueError(f"Invalid resource type: {resource_type}")




    