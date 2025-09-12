from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

document_analysis_prompt = ChatPromptTemplate.from_template(
    """
    you are highly capable assistant trained to analyze and summarize the documents.
    Return Only valid json matching the extract the schema below.

    {format_instructions}

    Analyze the document:

    {document_text}
    """
)
document_comparison_prompt = ChatPromptTemplate.from_template("""
you will be provided with content of two documents. The task are as follows:
1. compare the content of documents. 
2. Identify the differences in PDF and note down the page number. 
3. The output you provide should be page wise comparison content. 
4. if any page do not have any change, mention as "No changes".

input_documents:

{combined_docs}

your response should be in the below format:

{format_instructions}
""")


# Prompt for contextual question rewriting
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query as a standalone question "
        "that makes sense without relying on the previous context. Do not provide an answer—only reformulate the "
        "question if necessary; otherwise, return it unchanged."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Prompt for answering based on context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. Rely only on the retrieved "
        "information to form your response. If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
    "document_comparison": document_comparison_prompt,
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}