import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from graph.builder import graph

app = FastAPI(title="Trasklate API")


@app.post("/process")
async def process(text: str | None = Form(None), file: UploadFile | None = File(None)):
    """Send either `text` or an audio `file`; returns the routed model's answer."""
    if file:
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
        try:
            state = await graph.ainvoke({"audio_path": tmp.name})
        finally:
            os.unlink(tmp.name)
    elif text:
        state = await graph.ainvoke({"text": text})
    else:
        raise HTTPException(400, "Provide either `text` or an audio `file`.")

    return {k: state[k] for k in ("text", "intent", "model", "output")}
