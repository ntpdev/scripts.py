import os
# import openai
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from langchain_openai import ChatOpenAI
from textwrap import dedent


#os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key
#os.environ["OPENAI_API_KEY"] = "dummy"
#os.environ["OPENAI_API_BASE"]="http://localhost:1234/v1"
os.environ["OPENAI_MODEL_NAME"]="gpt-3.5-turbo"
# OPENAI_API_KEY = ''
#openai.api_key = 'dummy'
#openai.base_url = 'http://localhost:1234/v1'
llm = ChatOpenAI(model='gpt-3.5-turbo')

@tool("returns the result evaluating a mathematical expression.")
def calculate(expression: str) -> str:
    """returns the result evaluating a mathematical expression."""
    return eval(str).__repr__()

# manager = Agent(
#   role='Planner',
#   goal='give the task to the planner and then give the output to the reviewer. If the reviewer rejects the plan ask the planner to come up with a different option. Once the plan has passed review give it to the spokesman.',
#   verbose=True,
#   memory=True,
#   backstory=(
#     "You carefully follow instructions step by step."
#   ),
#   tools=[],
#   max_iter=5,
#   allow_delegation=True
# )

# Creating a senior researcher agent with memory and verbose mode
planner = Agent(
  role='Planner',
  goal='Answer the question in the problem statement.',
  verbose=True,
  memory=True,
  backstory=(
    "You carefully follow instructions step by step."
  ),
  tools=[],
  max_iter=5,
  allow_delegation=False
)

# Creating a writer agent with custom tools and delegation capability
reviewer = Agent(
  role='Quality assurance',
  goal='Do all QA tasks and say whether it is a pass or fail.',
  verbose=True,
  memory=True,
  backstory=(
    "You work step by step and independently check a plan. If the plan fails you should tell the planner that the list of coins was incorrect and to come up with another option. You do not need to suggest alternatives."
  ),
  tools=[],
  max_iter=5,
  allow_delegation=False
)

# Creating a writer agent with custom tools and delegation capability
# spokesman = Agent(
#   role='Spokesman',
#   goal='You give the final answer to the problem.',
#   verbose=True,
#   memory=True,
#   backstory=(
#     "You give the final answer to the problem. If it has not been solved you apologise."
#   ),
#   tools=[calculate],
#   max_iter=5,
#   allow_delegation=False
# )


# Research task
plan_task = Task(
  description=(dedent("""
  task pay for an {item} that costs 6 pence using the following list of available coins one 10p coin, one 5p coin, and three 2p coins.
  put your thinking within <thinking> tags. Choose coins that sum exactly to the cost of the item using only available coins.

  after you have done your thinking answer 'To pay for the {item} that costs 6p use' then a yaml list of the coins chosen.
  """)),
  expected_output='To pay for the {item} that costs 6p then a YAML list of the coins choosen.',
  tools=[],
 agent=planner,
)

# Writing task with language model configuration
review_task = Task(
  description=(dedent("""                                      
  Find the cost of the {item}. Find the list of coins that will be used to pay for it.
  Show the following information
    - cost of {item}
    - available coins                  
    - list of choosen coins
    - are all the choosen coins in the available list.
    - calculate sum of choosen coins
    - is the sum exactly equal to the cost of the {item}.
    - mark as either PASS or FAIL.
                      """)),
  expected_output='A pass / fail. If the plan passes repeat the list of coins selected.',
  tools=[],
 agent=reviewer,
)

# publish_task = Task(
#   description=(
#     "Tell the user the plan."
#   ),
#   expected_output='Explain to the user the coins they need to but the {item}.',
#   tools=[],
#   # agent=spokesman,
#   async_execution=False,
# )

# Forming the tech-focused crew with enhanced configurations
crew = Crew(
  agents=[planner, reviewer],
  tasks=[plan_task, review_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  # manager_llm=llm,
  verbose=True
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'item': 'apple'})
print(result)
