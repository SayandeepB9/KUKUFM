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
    def __init__(self, api_key=None, model_type="outline_generation"):
        self.llm = llm_api(api_key=api_key, model_type=model_type)
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

        self.refine_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", """Here is a previous story outline on the topic: {topic}
                
Previous events:
{previous_outline}

Human feedback:
{feedback}

Please refine the outline based on this feedback. Return a new list of events in the 'events' field."""),
            ]
        )
        self.refine_generator = self.refine_prompt | self.structured_llm_outline

    def generate_outline(self, topic):
        outline = self.outline_generator.invoke({"topic": topic})
        print("---GENERATED STORY OUTLINE---")
        for i, event in enumerate(outline.events, 1):
            print(f"Event {i}: {event}")
        return outline.events

    def refine_outline(self, topic, previous_events, feedback):
        """
        Refines a story outline based on human feedback.

        Args:
            topic (str): The original topic of the story
            previous_events (List[str]): The events from the previous outline
            feedback (str): Human feedback on how to improve the outline

        Returns:
            List[str]: Refined list of story events
        """
        previous_outline = "\n".join([f"{i}. {event}" for i, event in enumerate(previous_events, 1)])
        
        refined_outline = self.refine_generator.invoke({
            "topic": topic,
            "previous_outline": previous_outline,
            "feedback": feedback
        })
        
        print("---REFINED STORY OUTLINE---")
        for i, event in enumerate(refined_outline.events, 1):
            print(f"Event {i}: {event}")
            
        return refined_outline.events


if __name__ == "__main__":
    generator = OutlineGenerator()
    topic = "A horror story in a haunted hotel"
    events = generator.generate_outline(topic)
    print("Generated Outline:", events)
    
    # Example of using refine_outline
    feedback = "Make the story more spiritual and less about supernatural elements"
    refined_events = generator.refine_outline(topic, events, feedback)
    print("Refined Outline:", refined_events)