# mermaid_chart.py
import re

from topos.FC.ontological_feature_detection import OntologicalFeatureDetection
from topos.generations.chat_model import ChatModel


class MermaidChartGenerator:
    def __init__(self, chat_model: ChatModel):
        self.chat_model = chat_model

    def get_ontology_old_method(self, message):
        user_id = "jonny"
        session_id = "127348901"
        message_id = "1531ijasda8"

        composable_string = f"for user {user_id}, of {session_id}, the message is: {message}"

        ontological_feature_detection = OntologicalFeatureDetection("neo4j_uri", "neo4j_user", "neo4j_password",
                                                                    "showroom_db_name", False)

        entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities = ontological_feature_detection.build_ontology_from_paragraph(
            user_id, session_id, message_id, composable_string)

        input_components = message, entities, dependencies, relations, srl_results, timestamp, context_entities

        mermaid_syntax = ontological_feature_detection.extract_mermaid_syntax(input_components, input_type="components")
        mermaid_to_ascii = ontological_feature_detection.mermaid_to_ascii(mermaid_syntax)
        return mermaid_to_ascii

    @staticmethod
    def extract_mermaid_chart(response):
        mermaid_code = re.search(r'```mermaid\n(.*?)```', response, re.DOTALL)
        if mermaid_code:
            return mermaid_code.group(0)

        # Check for the variation with an extra newline character
        mermaid_code_variation = re.search(r'```\nmermaid\n(.*?)```', response, re.DOTALL)
        if mermaid_code_variation:
            print("\t[ reformatting mermaid chart ]")
            # Fix the variation by placing the mermaid text right after the ```
            fixed_mermaid_code = "```mermaid\n" + mermaid_code_variation.group(1) + "\n```"
            return fixed_mermaid_code
        return None

    @staticmethod
    def refine_mermaid_lines(mermaid_chart):
        lines = mermaid_chart.split('\n')
        refined_lines = []
        for line in lines:
            if '-->' in line:
                parts = line.split('-->')
                parts = [part.strip().replace(' ', '_') for part in parts]
                refined_line = ' --> '.join(parts)
                refined_lines.append("    " + refined_line)  # add the indent to the start of the line
            else:
                refined_lines.append(line)
        return '\n'.join(refined_lines)

    async def get_mermaid_chart(self, message, websocket=None):
        """
        Input: String Message
        Output: mermaid chart
        ``` mermaid
        graph TD
            Texas -->|is| hot
            hot -->|is| uncomfortable
            hot -->|is| unwanted
            Texas -->|actions| options
            options -->|best| Go_INSIDE
            options -->|second| Go_to_Canada
            options -->|last| try_not_to_die
        ```"""

        system_role = "Our goal is to help a visual learner better comprehend a sentence, by illustrating the text in a graph form. Your job is to create a list of graph triples from the speaker's sentence.\n"
        system_directive = """RULES:
        1. Extract graph triples from the sentence. 
        2. Use very simple synonyms to decrease the nuance in the statement. 
        3. Stay true to the sentence, make inferences about the sentiment, intent, if it is reasonable to do so.
        4. Use natural language to create the triples.
        5. Write only the comma separated triples format that follow node, relationship, node pattern
        6. If the statement is an opinion, create a relationship that assigns the speaker has_preference <object_of_preference>
        6. DO NOT HAVE ANY ISLAND RELATIONSHIPS. ALL EDGES MUST CONNECT."""
        system_examples = """```<examples>
        INPUT SENTENCE: The Texas heat is OPPRESSIVE
        OUTPUT:
        Texas, is, hot 
        hot, is, uncomfortable
        hot, is, unwanted
        ---
        SENTENCE: "Isn't Italy a better country than Spain?"
        OUTPUT:
        Italy, is_a, country
        Spain, is_a, country
        Italy, better, Spain
        better, property, comparison
        speaker, has_preference, Italy
        ```"""
        prompt = f"For the sake of illumination, represent this speaker's sentence in triples: {message}"
        system_ctx = system_role + system_directive + system_examples
        print("\t[ generating sentence_abstractive_graph_triples ]")
        if websocket:
            await websocket.send_json(
                {"status": "generating", "response": "generating sentence_abstractive_graph_triples",
                 'completed': False})
        sentence_abstractive_graph_triples = self.chat_model.generate_response(system_ctx, prompt)
        # print(sentence_abstractive_graph_triples)

        prompt = f"We were just given us the above triples to represent this message: '{message}'. Improve and correct their triples in a plaintext codeblock."
        print("\t[ generating refined_abstractive_graph_triples ]")
        if websocket:
            await websocket.send_json(
                {"status": "generating", "response": "generating refined_abstractive_graph_triples",
                 'completed': False})
        refined_abstractive_graph_triples = self.chat_model.generate_response(sentence_abstractive_graph_triples,
                                                                              prompt)  # a second pass to refine the first generation's responses
        # what is being said,

        # add relations to this existing graph that offer actions that can be taken, be humorous and absurd

        # output these graph relations into a mermaid chart we can use in markdown. Follow this form
        system_ctx = f"""Generate a mermaid block based off the triples.
        It should look like this:
    Example 1:
        ```mermaid
    graph TD;
        Italy--> |is_a| country;
        Spain--> |is_a| country;
        Italy--> better-->Spain;
        better-->property-->comparison;
        speaker-->has_preference-->Italy;
    ```
    Example 2:
    ```mermaid
    graph TD;
        High_School-->duration_of_study-->10_Years;
        High_School-->compared_to-->4_Year_Program;
        10_Year_Program-->more_time-->4_Years;
        Speaker-->seeks_change-->High_School_Length;
    ```
    Rules:
    1. No spaces between entities!
        """
        prompt = f"""Create a mermaid chart from these triples: {refined_abstractive_graph_triples}. Reduce the noise and combine elements if they are referencing the same thing. 
    Since United_States and The_United_States are the same thing, make the output just use: United_States.
    Example:
    Input triples
    ```
    United_States, has_spending, too_much
    The_United_States, could_do_with, less_spending;
    too_much, is, undesirable;
    ```

    Output Mermaid Where you Substitute The_United_States, with United_States.
    ```mermaid
    graph TD;
        United_States --> |has_spending| too_much;
        United_States --> |could_do_with| less_spending;
        too_much --> |is| undesirable;
    ```
    """
        attempt = 0
        message_history = [{"role": "system", "content": system_ctx}, {"role": "user", "content": prompt}]
        while attempt < 3:
            if attempt == 0:
                print("\t\t[ generating mermaid chart ]")
            if websocket:
                await websocket.send_json(
                    {"status": "generating", "response": "generating mermaid_chart_from_triples", 'completed': False})
            else:
                print(f"\t\t[ generating mermaid chart :: try {attempt + 1}]")
                if websocket:
                    await websocket.send_json({"status": "generating",
                                               "response": f"generating mermaid_chart_from_triples :: try {attempt + 1}",
                                               'completed': False})
            response = self.chat_model.generate_response_messages(message_history)
            mermaid_chart = self.extract_mermaid_chart(response)
            if mermaid_chart:
                # refined_mermaid_chart = refine_mermaid_lines(mermaid_chart)
                return mermaid_chart
            # print("FAILED:\n", response)
            message_history.append({"role": "user", "content": "That wasn't correct. State why and do it better."})
            attempt += 1
        return "Failed to generate mermaid"

# message = "Why can't we go to High School for 10 years instead of 4!!!"
# message2 = "Isn't Italy a better country than Spain?"
# message3 = "The United States could do with a little less spending"
# message4 = "As a group creating a product, we should be steady, have a clear view of the future, and not appear to succumb to dynamic market forces. If their argument takes over ours, they then argue that our organization's valuation could be nothing tomorrow because a new yet-to-be-made (ghost) tech will eat us up. Then they give us no money."
# message5 = "Furthermore, reading improves vocabulary and language skills, which is not as effectively achieved through watching TV."
# syntax = get_mermaid_chart(message4)
# print("MERMAID_CHART:\n", syntax)