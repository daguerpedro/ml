## Listas
```python
lista = ['A','B','C']
```
## Slice
```python
lista[inicio:fim]
lista[0:1] # 'A', 'B'
lista[:1] # 'A', 'B'
lista[:-1] # 'A', 'B'
lista[-1:] # 'B', 'C'
```
## Funções listas:
```python
len(lista)
sorted(lista)
sum(lista)
max(lista)
min(lista)
lista.append()
lista.pop()
lista.index()
```
## Comprehensions
```python
squares = [n**2 for n in range(10)]
squares
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

short_planets = [planet for planet in planets if len(planet) < 6]

```