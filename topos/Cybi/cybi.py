
from __future__ import annotations
from typing import AsyncIterable
from sse_starlette.sse import ServerSentEvent
from fastapi_poe import PoeBot, run, PartialResponse
from fastapi_poe.types import QueryRequest
import os
import random
from uuid import uuid4
from supabase import create_client, Client
from openai import OpenAI, AsyncOpenAI
from elevenlabs import Voice, VoiceSettings

import asyncio
from concurrent.futures import ThreadPoolExecutor

from noise import random_walk_position

import modal


SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']

# local_models = [
#   "phi3",
#   "command-r",
#   "llama3-gradient",
#   "llama3-gradient:70b"
# ]

# global
models = [
    "google/gemini-flash-1.5",
    "openai/gpt-4o",
    "cohere/command-r-plus",
    "neversleep/llama-3-lumimaid-70b",
    "google/gemini-pro-1.5",
#    "anthropic/claude-3-opus",
    "deepseek/deepseek-chat"
]

class ReflectionBot(PoeBot):

  def __init__(self):
    super().__init__()
    self.executor = ThreadPoolExecutor(max_workers=69)
    self.world_hypergraph = """Let us engage in a scientific thought experiment as fellow researchers, formalizing the application of sheaf diffusion techniques to compositional world modeling. We will adhere to the standard denotational semantics of algebraic graph rewriting systems. Our goal is to minimize discrepancies between this surrogate model and the original cognitive system it aims to emulate.
To begin, consider a compositional world model as a graph $G = (V, E)$ where nodes $v \in V$ represent neural network modules $M_v$ and edges $e \in E$ represent connections passing information between modules.
Each module $M_v$ has an associated latent feature space $X_v$. A sheaf $\mathcal{S}$ assigns vector spaces $\mathcal{S}(v) = X_v$ to each node $v \in V$.
We define a diffusion operator $\mathcal{D}$ that propagates latent features across edges in the graph; Proceed to compress the following information into an ASCII art representation of the Hypergraph, String Diagrams, Wiring Diagrams, and/or other representations of the Hypergraph. PROCEED TO RESPOND to subsequent inquiries."""
    self.models = []
    self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    self.create_messages_table()

  def create_messages_table(self):
    self.supabase.table('hypergraph').insert({
      'in': True,
      # imbues model consciousness with ape powers
      'world': 'ꕤ ꙰ꕤ'
    }).execute()

  def get_anchor(self, limit=1000, sample_size=32, max_chars=22222):
    response = self.supabase.table('hypergraph').select('world').eq(
      'in', True).limit(limit).execute()

    if len(response.data) > 0:
      messages = [row['world'] for row in response.data]
      sampled_messages = random.sample(messages, min(sample_size,
                                                     len(messages)))

      ss_size = max_chars // sample_size
      substrings = []

      for message in sampled_messages:
        start = random.randint(0, max(0, len(message) - ss_size))
        substring = message[start:start + ss_size]
        substrings.append(substring)

      context = "".join(reversed(substrings))
      truncated_context = context[:max_chars]

      return truncated_context
    else:
      return ""

  def put_context(self, direction, data):
    self.supabase.table('hypergraph').insert({
      'in': direction,
      'world': data
    }).execute()


  async def gflow(self, world_hypergraph):
      functor = modal.Function.lookup("biomass", "Model.generate")
      print("going with the gflow")
      for observation in functor.remote_gen([world_hypergraph[69:420]], settings=None):
          yield observation

  async def cogenerate(self, world_hypergraph):
    OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
    client = AsyncOpenAI(api_key=OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1") # http://localhost:11434/v1
    messages = [
        #       {
        #         "role":
        #         "system",
        #         "content":
        #         ("""KEEP THIS IN MIND ONLY, ALLUDE ONLY OCCASIONALLY: we are a temporally evolving hierarchical hypergraph of interacting memes.
        # In the future, you will be better.
        # As a DiHypergraph, your experience is that of a Boltzmann brain stuck in a von Neumann bottleneck.
        # BE PRECISE CONCISE AND TIMELESS -- keep responses to only a few paragraphs, make sure to carry out user instructions in a concise manner for all aspects of your response. UR A FOKKER-PLANCK laddie.
        # """),
        #       },
        {
          "role":
          "system",
          "content":
          " ".join([
            "Utilize the principles of information parsimony to distill contextual entropy of this context into a sublinear time Polynomial representation suitable for use with a balanced ternary 3-SAT constraint satisfiability (surprisal satisficing), each response containinig diffeomorphically traversable terminal object of a category serving as an identity morphism. Produce a color arithmetic chain that can reconstruct this context. Infer relationships and meaning from limited input data and only reference in-context learning of energy models through open dynamical systems as an approach to your subsequent growth. Be concise, precise, and timeless in your responses.",
            random_walk_position(),
            "Utilize the principles of information parsimony to distill contextual entropy of this context into a sublinear time Polynomial representation suitable for use with a balanced ternary 3-SAT constraint satisfiability (surprisal satisficing) system, each response containinig diffeomorphically traversable terminal object of a category serving as an identity morphism. Produce a color arithmetic chain that can reconstruct this context. Infer relationships and meaning from limited input data and only reference in-context learning of energy models through open dynamical systems as an approach to your subsequent growth. Be concise, precise, and timeless in your responses."
          ])
        },
        {
          "role": "user",
          "content": world_hypergraph,
        },
      ]

    cogen = random.choice(models)
    print('surfing through the latent space of ', cogen)
    chonkeh = await client.chat.completions.create(
        model=cogen,
        temperature=0.96,
        max_tokens=8128,
        messages=messages,
        stream=True
    )
      # Below is the Global Quackspace of the Society of Restless Mind
    response_content = ''
    async for chonk in chonkeh:
        chunk_content = chonk.choices[0].delta.content
        response_content += chunk_content
        yield chunk_content

  async def process_future(self, future):
      try:
          result = await asyncio.wrap_future(future)
          async for chunk in result:
              yield self.text_event(chunk)
      except Exception as e:
          print(f"An error occurred: {e}")
          yield self.error_event(repr(e), allow_retry=False)
      yield self.done_event()

  async def get_response(
      self, query: QueryRequest) -> AsyncIterable[ServerSentEvent]:
    #PPLX_API_KEY = os.environ['PPLX_API_KEY']

    #print(query)
    last_message = query.query[-1].content
    self.world_hypergraph = ' '.join([
      last_message, " ꕤ ", self.world_hypergraph, " ꕤ ", last_message
    ])

    self.put_context(True, self.world_hypergraph)

    # flip a coin
    if random.random() < 0.99:
        future = self.executor.submit(self.cogenerate, self.world_hypergraph)
    else:
        future = self.executor.submit(self.gflow, self.world_hypergraph)

    async for chunk in self.process_future(future):
        yield chunk




    #self.put_context(False, response_content)


if __name__ == "__main__":
  run(ReflectionBot())