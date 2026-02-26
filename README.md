## Clinical Assistant RAG Agent

Retrieval Augmented Generation API for clinical data using FastAPI, FAISS and Sentence-Transformers.

## Features: 
1.Semantic search over clinical data with FAISS
2. REST API endpoint built with FastAPI
3. Dockerized for easy deployment

## Architecture 
Clinical Note
     ↓
LLM Classification
     ↓
Structured Labels
     ↓
FastAPI Endpoints
     ↓
Query Engine

## Tech stack:
# PyTorch 2.2
# HuggingFace, Transformers
# FastAPI
# Uvicorn
# FAISS
# Python 3.11 

## Structure
# main.py -> FastAPI app
# retrieval.py -> FAISS index building script
