














# Step 9: FastAPI Interface
app = FastAPI()


@app.get("/generate_report/")
async def generate_report():
    prompt = "How many user do you have"
    response = report_chain.generate_report(prompt)
    return response

@app.post("/submit_feedback/")
async def submit_feedback(query: str, feedback: str, sql_query: str):
    report_chain.store_feedback(query, feedback, sql_query)
    return {"message": "Feedback submitted successfully"}

# To run the server: `uvicorn <filename_without_extension>:app --reload`
