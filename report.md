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

Jag bestämde att göra handlingarna diskreta för att minska arbetsmängden och för att det blir mer användbart för Monte Carlo metoden. 

---

## DQN

- uppskattar Q-värdet för framtida handlingar vid ett visst tillstånd med ett neuralt nätverk
- kör flera batches i taget för att uppskatta värdet



Jag följde AI's förslag på parametrarna lägsta och högsta värden.


---


<img src="images/" alt="Screenshot of " width="600"/>
