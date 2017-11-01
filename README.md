## Requirements
- numpy

## How to use

``` pip install -r requirements.txt ```

## import library

```python

from perceptron import Perceptron 

data = [[[0.25,0.25],-1],[[1,0.5],1],[[0.5,0.25],-1],[[0.25,1],1]]

bobot = [0,0],0

per = Perceptron(
    alpha=1,
    bobot=bobot, 
    data=data, 
    treshold=0
)

per.execute()


```


