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
        self.system_prompt = """You are a master storyteller in the tradition of classical Indian literature.
        Given a topic, generate a list of the main events that will happen in the story, drawing inspiration from rich Indian storytelling traditions like the Panchatantra, Jataka Tales, and ancient epics.
        
        For each story, you should provide 7-10 key events that form a coherent narrative with a clear beginning, middle, and end.
        Each event should be a detailed description of a significant plot point or development, incorporating cultural elements and wisdom.
        
        Your outline should include:
        - A compelling introduction that sets the scene and introduces the main themes
        - A series of events that build tension and develop characters
        - Challenges or conflicts that the characters must overcome
        - Resolution of conflicts with moral or ethical insights
        - A satisfying conclusion that delivers on the story's premise
        
        Create a narrative arc that follows classical storytelling with cultural authenticity and depth.
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

    def refine_outline(self, topic, outline, feedback):
        """
        Refine an existing outline based on feedback.
        
        Args:
            topic (str): The topic of the story
            outline (List[str]): The existing outline events
            feedback (str): Feedback to incorporate into the refined outline
            
        Returns:
            List[str]: The refined list of events
        """
        refine_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", """I have an outline for a story on the following topic: {topic}.
                
                Current outline:
                {outline}
                
                Feedback on the outline:
                {feedback}
                
                Please provide a refined outline that incorporates this feedback. Return a list of events in the 'events' field."""),
            ]
        )
        
        refine_chain = refine_prompt | self.structured_llm_outline
        
        # Format the outline as a numbered list for the prompt
        formatted_outline = "\n".join([f"{i+1}. {event}" for i, event in enumerate(outline)])
        
        refined_outline = refine_chain.invoke({
            "topic": topic,
            "outline": formatted_outline,
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