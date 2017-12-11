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

## LVQ Example
```python

from methods import LVQ

if __name__ == '__main__':
    vektor = [
        [[0,1,1,0], 1],
        [[0,0,1,1], 2],
        [[1,1,1,1], 1],
        [[1,0,0,1], 2]
    ]

    weight = [
        [[1,1,1,0],1],
        [[1,0,1,1], 2]
    ]

    alpha = 0.05
    dec_alpha = 0.1

    lv = LVQ(vektor, weight, alpha, dec_alpha)

    new_weight = lv.train(epoh=1)

    vektor_test = [
        [[1,1,1,0], 1],
        [[1,0,1,1], 1],
        
    ]

    print lv.test(vektor_test, new_weight)


```