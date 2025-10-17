# Kursprojekt - Rapport

*Viktor Nylund - Preskriptiv analytik*

“Method 1 is DQN with learned Q-function using CNN and automatic optimization.
Method 2 is a handcrafted Q-table approximation using a small, discrete action set, inspired by classical Q-learning from Assignment 3.”

## Innehåll

- [Process](#process)
 - [Innehållsbaserad filtreringen](#innehållsbaserad-filtreringen)
 - [Kollaborativ filtrering](#kollaborativ-filtrering)
 - [Hybrid rekommendation](#hybrid-rekommendation)
- [Resultat](#resultat)
- [Evaluering](#evaluering)

## Process

Jag kommer använda mig av  
**DQN** och **Monte Carlo**  
för att lösa Car Racing miljön.  
[Car Racing - Gymnasium](https://gymnasium.farama.org/environments/box2d/car_racing/)

Att byta till CarRacing-v3 från CartPole orsakade en del strul med versioner, men jag skapade en ny conda environment och hittade en kombination av versioner som fungerade tillsammans. De är specificerade i *requirements.txt* och *environment.yml*

Att byta till diskreta handlingar visade sig vara mycket lättare än förväntat. Jag önskar jag hade noterat det tidigare att det endast krävde `continuous=False`. 



---

## DQN

- uppskattar Q-värdet för framtida handlingar vid ett visst tillstånd med ett neuralt nätverk
- kör flera batches i taget för att uppskatta värdet


De bästa hyperparametrarna hittas av Optuna på samma vis som i uppgift *6 - Cartpole*.
Jag följde AI's förslag på parametrarna lägsta och högsta värden.


---


<img src="images/" alt="Screenshot of " width="600"/>
