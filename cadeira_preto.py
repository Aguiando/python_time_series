## complete a atividade abaixo, não esqueça de colocar o cabeçalho no arquivo.

import numpy as np
from itertools import combinations

EPS = 1e-9

#  Modele A, b, c
# Forma padrão: A @ [x,y] <= b, com não-negatividade via -I @ [x,y] <= 0
# Restrição 1: 2x +  y <= 100
# Restrição 2:  x + 2y <= 80
# Restrição 3:  x >= 0  ->  -x <= 0
# Restrição 4:  y >= 0  ->  -y <= 0
A = np.array([
    # preencha aqui
], dtype=float)

b = np.array([
    # preencha aqui
], dtype=float)

# Função objetivo: z = 3x + 5y
c = np.array([
    # preencha aqui
], dtype=float)


#  Interseção de duas retas
# Cada reta é dada por (a, beta) representando a @ [x,y] = beta
def intersecao(reta1, reta2):
    a1, b1 = reta1
    a2, b2 = reta2
    Aeq = np.vstack([a1, a2])        # 2x2
    beq = np.array([b1, b2], float)  # 2,
    # Se forem paralelas/singulares, retorne None
    # Dica: use det ou tente resolver e capture exceção
    # TODO:
    raise NotImplementedError


#  Resolver PL 2D via pontos extremos 
def resolve_lp_2d(A, b, c):
    m, n = A.shape
    assert n == 2, "Este resolvedor assume 2 variáveis (x, y)."

    linhas = [(A[i], b[i]) for i in range(m)]
    candidatos = []

    # Interseções de todos os pares de restrições (como igualdades)
    for i, j in combinations(range(m), 2):
        p = intersecao(linhas[i], linhas[j])
        if p is None:
            continue
        # Filtra viabilidade: A @ p <= b + EPS
        if np.all(A @ p <= b + EPS):
            candidatos.append(p)

    # Garante origem como fallback (útil se for viável)
    candidatos.append(np.array([0.0, 0.0]))

    P = np.array(candidatos)
    valores = P @ c
    k = int(np.argmax(valores))
    return P[k], float(valores[k]), P, valores
