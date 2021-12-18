from _ml import MLAgent, train, save, load, train_and_plot, RandomAgent, validate, plot_validation
from _core import is_winner, opponent, start
import random


 
 
class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward
    

random.seed(1)

my_agent = MyAgent(alpha=0.9, epsilon=0.3)

#Hyperparameters. Hyperparameters zorgen ervoor dat je de Agent anders kan samenstellen. Als het ware pas je de samenstelling van een functie ermee aan. Er zijn twee hyperparameters in functie van MyAgent, alfa en epsilon:

#Alfa is belangrijk om aan te kunnen geven hoe snel de agent oude kennis vervangt door nieuwe kennis. Het is dus de "leerfactor" van de Agent (zie fundament, 2.4). Hoe dichter het getal bij 1 zit, hoe sneller de nieuwe kennis overgenomen wordt.

#Epsilon zorgt ervoor dat de Agent nieuwe zetten blijft proberen en niet steeds bij de oude zetten met de beste uitkomsten blijft. Hij doet steeds vaker een random zet als het getal dicht bij 1 zit. Het kan dus zijn dat deze random zet juist beter is dan de oude best bekende zet, de winning-rate wordt dan hoger. Als de random zet juist slechter is wordt de winning rate weer lager. 

random_agent = RandomAgent()

train_and_plot(
	 agent=my_agent,
	 validation_agent=random_agent,
	 iterations=25,
	 trainings=200,
	 validations=1500)


train(my_agent, 3000)
 
save(my_agent, 'MyAgent_3000')

my_agent = load('MyAgent_3000')
 
my_agent.learning = True

validation_agent = RandomAgent()
 
validation_result = validate(agent_x=my_agent, agent_o=validation_agent, iterations=200)

plot_validation(validation_result)
 
start(player_x=my_agent)

