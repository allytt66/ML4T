
# Create testproject.py. Testproject.py is the entry point to your project, and it should implement the necessary calls (following each respective API) to Manual Strategy.py, StrategyLearner.py, experiment1.py, and experiment2.py with the appropriate parameters to run everything needed for the report in a single Python call: 


from experiment1 import experiment1
from experiment2 import experiment2

def author():
    return "yzhang3946"

if __name__ == '__main__': 
    experiment1()
    experiment2()