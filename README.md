# QuestMLProblemSets
Problem set answers for Fall Block 1 Machine Learning course at Quest University Canada

## Problem Set 2:
We have the code along with the final plot output of a single run, with the line ax_1+bx_2+c=0, where (using variable finder) we see that the weights are,

a = -0.0065419563183276574  
b = -0.0071808969876720526  
c = 0.007811730018380093

## Problem Set 3:
### k Nearest Neighbour:  
Letting k=2 and with the points (0.8, 0.2), (0.55,02), (0.2, 0.4) we get them classified as 0, 0, 1 respectively.

### Decision Trees:
I used the old titanic_medium dataset here to code a decision tree. We are able to construct the full decision tree with just two levels as the largest info gain at the start is gender, and the classes are exclusive of one another. We are therefore able to construct a decision tree of the following structure,

https://mermaidjs.github.io/mermaid-live-editor/#/view/eyJjb2RlIjoiZ3JhcGggVERcbkEoRmVtYWxlPykgLS0-fHl8IEIoQ2xhc3MxPylcbkEgLS0-fG58IEMoQ2xhc3MyPylcbkItLT58eXxEKC4uLilcbkItLT58bnxFKENsYXNzMz8pXG5DLS0-fHl8RiguLi4pXG5DLS0-fG58RyhDbGFzczM_KVxuRS0tPnx5fEgoLi4uKVxuRS0tPnxufEkoQ2xhc3MyKVxuRy0tPnx5fEooLi4uKVxuRy0tPnxufEsoQ2xhc3MxKVxuIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifX0

where (...) means an uncalculated probability of the person surviving.
