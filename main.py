from fastapi import FastAPI

from parlai.core.agents import create_agent_from_model_file


app = FastAPI()
blender_agent = None


def model_init():
    global blender_agent
    # import model from the model file can be pretrained or fine tuned
    blender_agent = create_agent_from_model_file("zoo:blender/blender_90M/model")


@app.on_event("startup")
def startup_event():
    print("Initializing model...")
    model_init()

@app.get("/response/{text}")
async def chatbot_response(text: str):
    blender_agent.observe({'text': text, 'episode_done': False})
    response = blender_agent.act()
  
    return {"response": response['text']}