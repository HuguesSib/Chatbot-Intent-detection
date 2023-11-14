# Chatbot-Intent-detection

The objective of this project is to create a chatbot that detect the intention of an input verbatim (prompt), for specific classes and in the presence of 'out of scope' prompts. The latter is outputed when the prompts doesn't belong to any given classes. 

The different classes are the following

- *translate*
- *travel_alert*
- *flight_status*
- *lost_luggage* : higher cost !
- *travel_suggestion*
- *carry_on*
- *book_hotel*
- *book_flight*

## Dataset 

A csv file was given with some examples (in french) of some prompts and their classes. Below is a distribution of this dataset. 

![alt text](/img/class_distib.png)

This set of prompt is too small to train a model on. We need to find a bigger model for this task. 



