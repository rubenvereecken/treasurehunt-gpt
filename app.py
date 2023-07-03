from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum
from langchain.schema import OutputParserException

class UserMessage(BaseModel):
    message: str
    roomSlug: str
    username: str
    ismod: bool

app = FastAPI()
handler = Mangum(app)

origins = [
    "http://localhost:3000",
    "https://ruben30.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def hello_world():
    return {'message': 'Hello World'}

@app.get("/")
def root():
    return {'message': 'This is the root mate'}


from treasurehunt import DynamoStateMachineMessageHistory, StateMachineBufferMemory, create_treasure_hunt_bot, get_riddles_str


@app.post("/huntgpt")
def hello(message: UserMessage):
    print('got message', message)
    
    try:
        riddles_str = get_riddles_str(message.roomSlug)    
    
        message_history = DynamoStateMachineMessageHistory(table_name="SessionTable-dev-local", session_id=message.roomSlug)
        message_memory = StateMachineBufferMemory(chat_memory=message_history)
        
        bot = create_treasure_hunt_bot(riddles_str=riddles_str, memory=message_memory)
        
        params = {
            'input': message.message,
            'user_role': 'Moderator' if message.ismod else 'Player'
        }
            
        # Retry 3 times
        for i in range(3):
            try: 
                out = bot(params)
                reply = out['reply']
                return {'reply': reply}
            except OutputParserException as e:
                continue
            except Exception as e:
                print(e)
                return {'reply': "Error getting bot response –" + str(e)[:200]}

    except Exception as e:
        print(e)
        return {'reply': "Error setting up bot –" + str(e)[:200]}
    