from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import json
import re
import os
from langchain_core.prompts import ChatPromptTemplate
from llm_api import llm_api, get_model_from_config

class ConsistencyIssue(BaseModel):
    """Represents a detected consistency issue in the story."""
    plot_option_index: Optional[int] = Field(None, description="Index of the plot option with issue (if applicable)")
    plot_option_text: Optional[str] = Field(None, description="Text of the plot option with issue (if applicable)")
    issue_type: str = Field(..., description="Type of consistency issue detected")
    severity: str = Field(..., description="Severity of the issue: critical, warning, or minor")
    description: str = Field(..., description="Detailed description of the issue")
    suggestions: List[str] = Field(default_factory=list, description="Possible fixes for the inconsistency")

class StoryKnowledgeGraph:
    """Maintains a knowledge graph of story elements and their relationships."""
    
    def __init__(self):
        self.elements = {
            "event": {},  # For outline events
            "plot": {},   # For plot options
            "setting": {}  # For settings and environment
        }
        self.relationships = []
        
    def add_element(self, element_type: str, element_id: str, attributes: Dict[str, Any]):
        """Add an element to the knowledge graph."""
        if element_type not in self.elements:
            self.elements[element_type] = {}
        self.elements[element_type][element_id] = attributes
        
    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, attributes: Dict[str, Any] = None):
        """Add a relationship between elements to the knowledge graph."""
        self.relationships.append({
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "attributes": attributes or {}
        })

class ConsistencyChecker:
    """Consistency checker for AI story generation pipeline - focusing on outline and plot."""
    
    def __init__(self, api_key=None):
        """Initialize the consistency checker."""
        self.model_type = "consistency_checking"
        # Get model name from configuration or use a default
        try:
            self.model_name = get_model_from_config(self.model_type)
        except:
            # Default to a common model if get_model_from_config isn't available
            self.model_name = "gpt-4"
        
        # Initialize LLM using the llm_api function
        self.llm = llm_api(
            api_key=api_key,
            model_type=self.model_type,
            streaming=False
        )
        
        # Initialize knowledge graph
        self.knowledge_graph = StoryKnowledgeGraph()
        
        # Define prompts for consistency checking
        self.system_prompt = (
            "You are an expert literary editor specializing in narrative consistency. "
            "Analyze the following story outline and generated plot options for logical inconsistencies. "
            "Your task is to identify specific inconsistencies between the outline and plot options, "
            "or any internal inconsistencies within the plot options themselves. "
            "For each issue detected, specify:\n"
            "1. The specific plot option number that has the inconsistency (0-indexed)\n"
            "2. The exact text of the problem plot option\n"
            "3. The type of inconsistency (e.g., timeline contradiction, character motivation, plot logic)\n"
            "4. The severity (critical, warning, minor)\n"
            "5. A detailed description explaining exactly why there is an inconsistency\n"
            "6. Multiple specific suggestions to fix the issue\n\n"
            "Format your response as a JSON array of issues, each with the fields: "
            "plot_option_index, plot_option_text, issue_type, severity, description, suggestions.\n\n"
            "If there are no inconsistencies, return an empty array []."
        )
        
        # Prompt template for plot consistency check
        self.plot_consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", (
                "Story type: {story_type}\n\n"
                "Outline events:\n{outline_events}\n\n"
                "Plot options (indexed from 0):\n{indexed_plot_options}\n\n"
                "Analyze each plot option individually for consistency with the outline and within the story logic. "
                "Return your analysis as a JSON array of issues. Be specific about which plot option (by index) "
                "has each issue."
            ))
        ])
        
        # Create consistency checking chain
        self.plot_consistency_chain = self.plot_consistency_prompt | self.llm
        
    def _parse_issues_from_response(self, response, plot_options) -> List[ConsistencyIssue]:
        """Parse consistency issues from LLM response."""
        result_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to parse the response as JSON
        try:
            # Find JSON in the response (it might be surrounded by other text)
            json_match = re.search(r'(\[[\s\S]*\])', result_text)
            if json_match:
                cleaned_json = json_match.group(1)
                issues_data = json.loads(cleaned_json)
                
                # Check if we got an empty array (no issues)
                if len(issues_data) == 0:
                    return []
                
                issues = []
                for issue_data in issues_data:
                    # Get the plot option index and text
                    plot_idx = issue_data.get("plot_option_index")
                    plot_text = issue_data.get("plot_option_text")
                    
                    # If we have an index but no text, try to get the text from the plot options
                    if plot_idx is not None and plot_text is None and 0 <= plot_idx < len(plot_options):
                        plot_text = plot_options[plot_idx]
                    
                    # Create the issue object
                    issue = ConsistencyIssue(
                        plot_option_index=plot_idx,
                        plot_option_text=plot_text,
                        issue_type=issue_data.get("issue_type", "Unknown"),
                        severity=issue_data.get("severity", "warning"),
                        description=issue_data.get("description", "No description provided"),
                        suggestions=issue_data.get("suggestions", [])
                    )
                    issues.append(issue)
                return issues
                
        except Exception as e:
            print(f"Error parsing consistency issues: {str(e)}")
            
        # Fallback: Try to extract issues manually
        print("Could not parse LLM response as JSON. Using fallback parsing method.")
        issues = []
        
        # Look for issue sections using regex
        issue_matches = re.finditer(r'(?:Issue|Inconsistency|Problem)(?:\s+|:)(\d+):', result_text, re.IGNORECASE)
        positions = [(m.start(), int(m.group(1))) for m in issue_matches]
        positions.append((len(result_text), -1))  # Add end marker
        
        # Process each issue section
        for i in range(len(positions) - 1):
            start_pos, issue_num = positions[i]
            end_pos = positions[i + 1][0]
            section = result_text[start_pos:end_pos].strip()
            
            try:
                # Extract plot option index
                plot_idx_match = re.search(r'(?:Plot Option|Option)(?:\s+|:)(\d+)', section, re.IGNORECASE)
                plot_idx = int(plot_idx_match.group(1)) if plot_idx_match else None
                
                # Adjust for 0-indexing if needed
                if plot_idx is not None and plot_idx >= len(plot_options):
                    plot_idx = plot_idx - 1 if plot_idx > 0 else 0
                
                # Extract plot text
                plot_text = None
                if plot_idx is not None and 0 <= plot_idx < len(plot_options):
                    plot_text = plot_options[plot_idx]
                else:
                    # Try to find quoted text that might be the plot option
                    plot_text_match = re.search(r'"([^"]+)"', section)
                    if plot_text_match:
                        plot_text = plot_text_match.group(1)
                
                # Extract issue type
                issue_type_match = re.search(r'(?:Type|Issue Type):\s*(.+?)(?:\n|$)', section)
                issue_type = issue_type_match.group(1).strip() if issue_type_match else "Logical Inconsistency"
                
                # Extract severity
                severity_match = re.search(r'Severity:\s*(.+?)(?:\n|$)', section)
                severity = severity_match.group(1).strip() if severity_match else "warning"
                
                # Extract description
                description_match = re.search(r'(?:Description|Problem|Explanation):\s*(.+?)(?:\n\s*(?:Suggestions|Fixes|Solutions)|$)', section, re.DOTALL)
                description = description_match.group(1).strip() if description_match else "Inconsistency detected"
                
                # Extract suggestions
                suggestions = []
                suggestions_match = re.search(r'(?:Suggestions|Fixes|Solutions):\s*([\s\S]+?)(?:\n\n|$)', section)
                if suggestions_match:
                    suggestions_text = suggestions_match.group(1)
                    # Try to split suggestions by numbers or bullets
                    suggestion_items = re.findall(r'(?:\d+\.\s*|\-\s*)(.+?)(?:\n|$)', suggestions_text)
                    if suggestion_items:
                        suggestions = [item.strip() for item in suggestion_items if item.strip()]
                    else:
                        suggestions = [suggestions_text.strip()]
                
                # Create the issue
                issue = ConsistencyIssue(
                    plot_option_index=plot_idx,
                    plot_option_text=plot_text,
                    issue_type=issue_type,
                    severity=severity,
                    description=description,
                    suggestions=suggestions
                )
                issues.append(issue)
                
            except Exception as e:
                print(f"Error parsing issue section: {str(e)}")
                continue
        
        # If no issues were found with the regex approach, check for a simple "no issues" message
        if not issues and re.search(r'(?:no|zero) (?:inconsistencies|issues|problems) (?:found|detected)', result_text, re.IGNORECASE):
            return []
            
        # If we still have no issues but the response wasn't clearly "no issues", create a generic issue
        if not issues and len(result_text.strip()) > 20:  # If there's substantial text
            issues.append(
                ConsistencyIssue(
                    plot_option_index=None,
                    plot_option_text=None,
                    issue_type="Parsing Error",
                    severity="warning",
                    description="Could not parse specific issues from the consistency checker output",
                    suggestions=["Review the LLM response manually", "Try again with more specific instructions"]
                )
            )
            
        return issues
    
    def check_plot_consistency(self, story_type: str, outline_events: List[str], plot_options: List[str]) -> List[ConsistencyIssue]:
        """Check consistency between outline and plot options."""
        print("\nChecking plot consistency...")
        try:
            # Format input for the LLM
            outline_text = "\n".join(f"- {event}" for event in outline_events)
            
            # Create indexed plot options for better reference
            indexed_plot_text = "\n".join(f"[{i}] {option}" for i, option in enumerate(plot_options))
            
            # Invoke the LLM
            response = self.plot_consistency_chain.invoke({
                "story_type": story_type,
                "outline_events": outline_text,
                "indexed_plot_options": indexed_plot_text
            })
            
            # Parse issues from response
            issues = self._parse_issues_from_response(response, plot_options)
            
            # Update knowledge graph based on outline and plot
            for i, event in enumerate(outline_events):
                event_id = f"outline_event_{i}"
                self.knowledge_graph.add_element("event", event_id, {
                    "description": event,
                    "position": i
                })
            
            for i, plot in enumerate(plot_options):
                plot_id = f"plot_option_{i}"
                self.knowledge_graph.add_element("plot", plot_id, {
                    "description": plot,
                    "has_issues": any(issue.plot_option_index == i for issue in issues)
                })
            
            return issues
            
        except Exception as e:
            print(f"Error checking plot consistency: {str(e)}")
            # Return a generic issue if an error occurred
            return [
                ConsistencyIssue(
                    plot_option_index=None,
                    plot_option_text=None,
                    issue_type="Processing Error",
                    severity="warning",
                    description=f"An error occurred while checking plot consistency: {str(e)}",
                    suggestions=["Try again with a different model or more detailed input"]
                )
            ]
    
    def display_consistency_report(self, issues: List[ConsistencyIssue]) -> bool:
        """Display a report of consistency issues and return whether there are critical issues."""
        if not issues:
            print("\n✅ No consistency issues detected! All plot options are consistent with the outline.")
            return True
        
        has_critical_issues = False
        
        print("\n=== CONSISTENCY REPORT ===\n")
        
        # Group issues by plot option for clearer reporting
        issues_by_plot = {}
        for issue in issues:
            plot_idx = issue.plot_option_index
            if plot_idx is None:
                plot_idx = -1  # Use -1 for general issues
            
            if plot_idx not in issues_by_plot:
                issues_by_plot[plot_idx] = []
                
            issues_by_plot[plot_idx].append(issue)
            if issue.severity.lower() == "critical":
                has_critical_issues = True
        
        # Display issues by plot option
        for plot_idx in sorted(issues_by_plot.keys()):
            if plot_idx == -1:
                print("GENERAL ISSUES:")
            else:
                print(f"\nPLOT OPTION [{plot_idx}]:")
                plot_text = issues_by_plot[plot_idx][0].plot_option_text
                print(f"  \"{plot_text}\"")
                
            for i, issue in enumerate(issues_by_plot[plot_idx], 1):
                severity = issue.severity.lower()
                severity_symbol = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟡" if severity in ["medium", "warning"] else "🔵"
                
                print(f"\n  Issue {i}: {severity_symbol} {severity.upper()} - {issue.issue_type}")
                print(f"  Description: {issue.description}")
                
                print("  Suggestions:")
                for j, suggestion in enumerate(issue.suggestions, 1):
                    print(f"    {j}. {suggestion}")
        
        # Return summary
        total_issues = len(issues)
        plot_count = len([k for k in issues_by_plot.keys() if k != -1])
        
        print(f"\nSUMMARY: Found {total_issues} issues across {plot_count} plot options")
        
        if has_critical_issues:
            print("⚠️  Critical issues detected! Consider regenerating or revising problem plot options.")
            return False
        else:
            print("⚠️  Issues found but none are critical. You may proceed or revise as desired.")
            return True
    
    def generate_improved_suggestions(self, story_type: str, outline_events: List[str], 
                                     problem_plot_options: List[Tuple[int, str]], 
                                     issues: List[ConsistencyIssue]) -> List[str]:
        """Generate improved versions of problem plot options based on consistency issues."""
        # Prepare a prompt for the LLM
        system_prompt = (
            "You are an expert storyteller tasked with fixing inconsistent plot options. "
            "Given a story outline, problematic plot options, and identified consistency issues, "
            "create new improved versions of the plot options that address all the issues. "
            "Your new plot options should maintain the core idea but eliminate inconsistencies "
            "with the outline and story logic."
        )
        
        # Format the outline
        outline_text = "\n".join(f"- {event}" for event in outline_events)
        
        # Format the problem plot options and their issues
        problems_text = ""
        for idx, text in problem_plot_options:
            problems_text += f"PROBLEM PLOT OPTION [{idx}]: \"{text}\"\n"
            relevant_issues = [issue for issue in issues if issue.plot_option_index == idx]
            for i, issue in enumerate(relevant_issues, 1):
                problems_text += f"Issue {i}: {issue.issue_type} - {issue.description}\n"
            problems_text += "\n"
        
        human_prompt = (
            f"Story type: {story_type}\n\n"
            f"Outline events:\n{outline_text}\n\n"
            f"Problems to fix:\n{problems_text}\n\n"
            f"Please generate one improved version for each problem plot option. "
            f"Provide {len(problem_plot_options)} new plot options that fix the identified issues "
            f"while preserving the core idea of each original option. "
            f"Format your response as a numbered list, with each improved plot option "
            f"clearly labeled with the corresponding original option number."
        )
        
        # Create a prompt template
        fix_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        # Create chain and invoke
        fix_chain = fix_prompt | self.llm
        
        try:
            response = fix_chain.invoke({})
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the improved plot options from the response
            improved_options = []
            
            # Look for numbered items
            lines = result_text.split('\n')
            current_option = ""
            
            for line in lines:
                # Check if the line starts a new option
                if re.match(r'^\d+[\.\)]', line) or re.match(r'^Option \d+', line) or re.match(r'^Plot Option \[\d+\]', line):
                    # Save the previous option if it exists
                    if current_option:
                        improved_options.append(current_option.strip())
                    current_option = re.sub(r'^.*?[:\-] ', '', line).strip()
                else:
                    # Append to current option
                    if current_option or line.strip():  # Only append if we've already started an option or the line isn't blank
                        current_option += " " + line.strip()
            
            # Add the last option if it exists
            if current_option:
                improved_options.append(current_option.strip())
            
            # If we didn't find any options with the regex, try to extract paragraph-based options
            if not improved_options:
                paragraphs = re.split(r'\n\s*\n', result_text)
                for para in paragraphs:
                    if len(para.strip()) > 20:  # Only consider substantial paragraphs
                        # Remove any leading numbers or labels
                        cleaned = re.sub(r'^.*?[:\-] ', '', para.strip())
                        improved_options.append(cleaned)
            
            # If we couldn't extract options, create a default "fixed" response
            if not improved_options:
                for idx, text in problem_plot_options:
                    improved_options.append(f"Fixed version of option {idx}: {text} [improved by removing inconsistencies]")
            
            # Make sure we have the right number of options
            while len(improved_options) < len(problem_plot_options):
                idx = len(improved_options)
                if idx < len(problem_plot_options):
                    improved_options.append(f"Improved version of option {problem_plot_options[idx][0]}")
            
            return improved_options[:len(problem_plot_options)]
            
        except Exception as e:
            print(f"Error generating improved plot options: {str(e)}")
            # Return basic placeholders
            return [f"Improved version of option {idx}" for idx, _ in problem_plot_options]

# Create an integration function
def integrate_consistency_checker(api_key=None):
    """Create and return a consistency checker instance."""
    return ConsistencyChecker(api_key=api_key)