# Kursprojekt - Rapport

*Viktor Nylund - Preskriptiv analytik*

## Innehåll

- [Process](#process)
 - [Innehållsbaserad filtreringen](#innehållsbaserad-filtreringen)
 - [Kollaborativ filtrering](#kollaborativ-filtrering)

- [DQN](#dqn)


## Process

Jag kommer använda mig av **DQN** och **Monte Carlo**  
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

| Parameter               | Värde                     |
|-------------------------|---------------------------|
| learning_rate           | 1.04969253444325e-05     |
| gamma                   | 0.9905143088423188       |
| layer_size              | 128                       |
| batch_size              | 64                        |
| tau                     | 0.037882606079686784     |
| exploration_final_eps   | 0.06054294571836032      |
| buffer_size             | 100000                    |
| exploration_fraction    | 0.24991722761412102      |
| target_update_interval  | 1000                      |
| train_freq              | 8                         |
| gradient_steps          | 1                         |
| Mean Reward             | 14.45                     |



---


<img src="images/" alt="Screenshot of " width="600"/>
