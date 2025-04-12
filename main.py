import sys
import os
import dotenv
import json
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pymilvus import MilvusException

# Load environment variables from .env
dotenv.load_dotenv()

# Initialize MilvusClient (assumes MilvusClient is defined in a separate module)
from MilvusClient import MilvusClient
milvus_client = MilvusClient()

# Configuration from environment variables
collection_name = os.getenv("MILVUS_COLLECTION")
TOP_K = int(os.getenv("TOP_K", 3))
OUTPUT_FIELDS = ["job_id", "title", "company_name", "required_skills"]

# Define a custom tool for Milvus hybrid search
class MilvusJobSearchTool(BaseTool):
    name: str = "Milvus Hybrid Job Search Tool"
    description: str = (
        "Searches a Milvus vector database using hybrid search (dense + sparse) "
        "for relevant job postings based on a list of skills. "
        "Input must be a comma-separated string of skills."
    )
    milvus_handler: MilvusClient = None  # To hold the client instance

    def __init__(self, milvus_handler: MilvusClient, **kwargs):
        super().__init__(**kwargs)
        self.milvus_handler = milvus_handler
        print(f"Initialized MilvusJobSearchTool with provided MilvusClient handler.")

    def _run(self, skills_list_str: str) -> str:
        if not self.milvus_handler:
            return "Error: MilvusClient handler was not provided during initialization."
        if not collection_name:
            return "Error: Milvus collection name not configured (MILVUS_COLLECTION)."

        print(f"Received skills for hybrid search: {skills_list_str}")
        print(f"Target collection: {collection_name}, Limit: {TOP_K}, Output Fields: {OUTPUT_FIELDS}")

        results = []
        try:
            print("Performing hybrid search via MilvusClient...")
            search_results = self.milvus_handler.hybrid_search(
                collection_name=collection_name,
                query_text=skills_list_str,
                limit=TOP_K,
                ranker_type="weighted",
                sparse_weight=0.3,
                dense_weight=0.7,
                output_fields=OUTPUT_FIELDS
            )

            print(f"Hybrid search completed. Found {len(search_results)} potential matches.")

            for hit in search_results:
                job_info = {"milvus_id": hit["id"]}
                for field in OUTPUT_FIELDS:
                    job_info[field] = hit["entity"].get(field, f"{field}_not_found")
                job_info["score"] = hit["distance"]
                results.append(job_info)
            print(f"Processed results: {results}")

        except MilvusException as e:
            print(f"Milvus search error via client: {e}")
            return f"Error during Milvus hybrid search: {e}"
        except AttributeError as e:
            print(f"AttributeError during search: {e}. Check MilvusClient method.")
            return f"Error: Missing required method in MilvusClient: {e}"
        except Exception as e:
            print(f"Unexpected error during hybrid search via client: {type(e).__name__} - {e}")
            return f"Unexpected error during Milvus hybrid search: {e}"

        if not results:
            return "No relevant jobs found in Milvus using hybrid search."
        else:
            return json.dumps(results, indent=2)

# Initialize LLM (using GPT-4o from OpenAI)
llm = LLM(
    model="gpt-4o",
    api_key=os.getenv('OPENAI_API_KEY'),
    max_tokens=2000,
    temperature=0
)

# Define Skill Extractor Agent
skill_extractor_agent = Agent(
    role='Skill Extractor',
    goal='Extract 5 to 10 skills from text as a comma-separated list.',
    backstory="",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Define Skill Extraction Task
extract_skills_task = Task(
    description=(
        "Carefully analyze the provided CV text: ```{cv_text}``` "
        "Identify and extract 5 to 10 technical skills, programming languages, frameworks, tools, and methodologies mentioned. "
        "Focus only on skills listed explicitly in the text. "
        "Your final output MUST be ONLY the comma-separated list of skills and nothing else."
    ),
    expected_output=(
        "A single string containing the identified skills, separated ONLY by commas. Do not include any other text, explanation, or formatting. Between 5 and 10 skills. "
        "Example: Python, Java, AWS, Docker, SQL, Agile, Problem Solving"
    ),
    agent=skill_extractor_agent,
)

# Instantiate Milvus Job Search Tool
milvus_tool = MilvusJobSearchTool(milvus_handler=milvus_client)

# Define Job Search Agent
job_search_agent = Agent(
    role='Milvus Job Search Specialist',
    goal=(
        f'Use the provided comma-separated list of skills to perform a HYBRID search for relevant job postings '
        f'in the Milvus vector database using the "{MilvusJobSearchTool(milvus_handler=milvus_client).name}".'
    ),
    backstory=(
        "Specialist in hybrid search within vector databases using the available tool "
        f"to find the top {TOP_K} most relevant job postings."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[milvus_tool],
    llm=llm,
)

# Define Job Search Task
search_jobs_task = Task(
    description=(
        f"Take the comma-separated list of skills identified in the previous step and use the "
        f'"{MilvusJobSearchTool(milvus_handler=milvus_client).name}" to find the top {TOP_K} relevant job postings from the Milvus database. '
        "The tool requires the skills as a single comma-separated string input."
    ),
    expected_output=(
        "A JSON formatted string list of the matching jobs found in Milvus, including their ID, title, "
        f"and similarity score, based on the output of the {MilvusJobSearchTool(milvus_handler=milvus_client).name}. "
        "If no jobs are found, state that clearly."
    ),
    agent=job_search_agent,
    context=[extract_skills_task],
)

# Define Match Explainer Agent
match_explainer_agent = Agent(
    role='Career Analyst and Job Match Explainer',
    goal=(
        "Analyze the provided list of matched jobs (in JSON format) and the candidate's CV text. "
        "Provide a concise, insightful explanation of why the top job matches are relevant to the candidate, "
        "referencing specific skills or experiences from the CV."
    ),
    backstory=(
        "You are an experienced career advisor with a keen eye for matching candidate profiles "
        "with suitable job opportunities. You excel at articulating the connection between a candidate's "
        "qualifications (from their CV) and the requirements suggested by job titles and descriptions."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm,
)

# Define Match Explanation Task
explain_matches_task = Task(
    description=(
        "You will be provided with a candidate's CV and the JSON results from a previous job search task.\n\n"
        "Candidate CV:\n"
        "```\n"
        "{cv_text}\n"
        "```\n\n"
        "Your task is to analyze both the CV and the JSON list of jobs (provided as context from the previous task). "
        "Based on the job titles, IDs, and potentially the text snippet (if available in the JSON), "
        "explain *why* the top 2-3 jobs listed are a good potential match for the candidate. "
        "Reference specific skills, experiences, or technologies mentioned in the CV to justify the relevance "
        "of each explained match. Focus on clear, concise reasoning."
    ),
    expected_output=(
        "A well-written, human-readable summary explaining the relevance of the top 2-3 job matches. "
        "This could be a short paragraph per match or a bulleted list. "
        "Example: \n"
        "- Job 'Data Analyst (ID: job123)' seems relevant because the CV highlights strong experience in Python, SQL, and Data Analysis, which are core skills for this role.\n"
        "- The 'Python Developer (ID: job456)' position aligns with the candidate's extensive background in Python frameworks like Flask and Django mentioned under 'Experience'."
    ),
    agent=match_explainer_agent,
    context=[search_jobs_task]
)

# Define the Job Matching Crew
job_matching_crew = Crew(
    agents=[skill_extractor_agent, job_search_agent, match_explainer_agent],
    tasks=[extract_skills_task, search_jobs_task, explain_matches_task],
    process=Process.sequential,
    verbose=True
)

# Reusable function to run the job matching crew
def run_job_matching(resume_text: str):
    """Run the job matching crew with the provided resume text and return the result."""
    inputs = {'cv_text': resume_text}
    result = job_matching_crew.kickoff(inputs=inputs)
    return result

# CLI execution (runs only when script is executed directly)
if __name__ == "__main__":
    # Default resume file or take from command line argument
    resume_filename = sys.argv[1] if len(sys.argv) > 1 else "Alex Chen Resume.md"
    with open(f"sample_resumes/{resume_filename}", "r") as file:
        resume = file.read()

    print("Starting main crew execution...")
    result = run_job_matching(resume)

    # Save results to a file
    os.makedirs("results", exist_ok=True)
    with open(f"results/job_matching_result_{resume_filename}", "w") as file:
        file.write(str(result))

    # Print results to console
    print('=' * 50)
    print(result)
    print('=' * 50)