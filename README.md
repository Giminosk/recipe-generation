# Recipe Generation

## Main idea

The goal is to introduce a model that would be capable of generating new recipes based on a user's input ingredients. 

Basing on Eat-PIM generated embeddings, the input user's ingredients would result in a FlowGraph, presenting the recipe proposed by the model.

## Chain of work

- Parse provided ingredients
- Match infringements with Foodon and Wikidata entities
- Generate a set of the most applicable actions for these ingredients
- Create a FlowGraph from these ingredients and actions
- Generate textual recipe based on the FlowGraph
