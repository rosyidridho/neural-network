## Requirements
- numpy

## How to use

``` pip install -r requirements.txt ```

## import library

```python

from methods import Perceptron 

data = [[[0.25,0.25],-1],[[1,0.5],1],[[0.5,0.25],-1],[[0.25,1],1]]

weight = [0,0],0

per = Perceptron(
    alpha=1,
    weight=weight, 
    dataset=data, 
    treshold=0
)

train = per.train()
test = per.test(data, get_new_weight(train))
percentage = per.get_percentage(test)

```


