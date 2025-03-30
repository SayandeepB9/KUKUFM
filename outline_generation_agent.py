from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api


class OutlineQuery(BaseModel):
    """Generate an outline for a story given a topic."""

    topic: str = Field(
        ...,
        description="The topic for which the outline of the story should be generated.",
    )
    events: List[str] = Field(
        default_factory=list,
        description="The main events that will happen in the story outline. Provide at least 5-7 events.",
    )


class OutlineGenerator:
    def __init__(self, api_key=None):
        self.llm = llm_api(api_key=api_key, model_type="outline_generation")
        self.system_prompt = """You are an expert at generating detailed story outlines.
        Given a topic, generate a list of the main events that will happen in the story.
        For each story, you should provide at least 5-7 key events that form a coherent narrative.
        Each event should be a brief description of a significant plot point or development.
        FORMAT YOUR RESPONSE AS A LIST OF EVENTS ONLY."""
        self.structured_llm_outline = self.llm.with_structured_output(OutlineQuery)
        self.outline_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "Generate an outline for a story on the following topic: {topic}. Return a list of events in the 'events' field."),
            ]
        )
        self.outline_generator = self.outline_prompt | self.structured_llm_outline

    def generate_outline(self, topic):
        outline = self.outline_generator.invoke({"topic": topic})
        print("---GENERATED STORY OUTLINE---")
        for i, event in enumerate(outline.events, 1):
            print(f"Event {i}: {event}")
        return outline.events


if __name__ == "__main__":
    generator = OutlineGenerator()
    topic = "A horror story in a haunted hotel"
    events = generator.generate_outline(topic)
    print("Generated Outline:", events)