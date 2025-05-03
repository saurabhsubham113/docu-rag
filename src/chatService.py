from openai import OpenAI

class ChatService:
    def __init__(self,model="gpt-4.1",temperature=0.3):
        """Initialize chat service with openAI client"""
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.conversastion_history = []

    def generate_response(self,system_prompt,user_input):
        """generate a response using the openAI"""
        try:
            self.conversastion_history.append({"role":"user","content":user_input})

            messages = [
                {"role":"system","content":system_prompt}
            ] + self.conversastion_history

            # get result from the openAI
            result = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages
            )

            response = result.choices[0].message.content

            # Add the response to the conversastion history
            self.conversastion_history.append({"role":"assistant","content":response})

            return response

        except Exception as e:
            print(f"error while generating openAI response {e}")

    def clear_chat_history(self):
        """clear all conversastions from the memory"""
        self.conversastion_history = []
