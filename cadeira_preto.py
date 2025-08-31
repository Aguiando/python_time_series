"""
Programação Linear 2D - Resolvedor por Pontos Extremos
Autor: Daniel de Souza de Aguiar
Data: 31/08/2025
Disciplina: Dinâmica de Evolução de Software
Professor: Fabiano Dicheti

Problema:
Maximizar z = 3x + 5y
Sujeito a:
  2x + y <= 100
  x + 2y <= 80
  x >= 0
  y >= 0
"""

import numpy as np
from itertools import combinations

EPS = 1e-9

# Modele A, b, c
# Forma padrão: A @ [x,y] <= b, com não-negatividade via -I @ [x,y] <= 0
# Restrição 1: 2x +  y <= 100
# Restrição 2:  x + 2y <= 80
# Restrição 3:  x >= 0  ->  -x <= 0
# Restrição 4:  y >= 0  ->  -y <= 0
A = np.array([
    [2, 1],   # 2x + y <= 100
    [1, 2],   # x + 2y <= 80
    [-1, 0],  # -x <= 0 (x >= 0)
    [0, -1]   # -y <= 0 (y >= 0)
], dtype=float)

b = np.array([
    100,  # <= 100
    80,   # <= 80
    0,    # <= 0 (para x >= 0)
    0     # <= 0 (para y >= 0)
], dtype=float)

# Função objetivo: z = 3x + 5y
c = np.array([
    3,  # coeficiente de x
    5   # coeficiente de y
], dtype=float)


# Interseção de duas retas
# Cada reta é dada por (a, beta) representando a @ [x,y] = beta
def intersecao(reta1, reta2):
    a1, b1 = reta1
    a2, b2 = reta2
    Aeq = np.vstack([a1, a2])        # 2x2
    beq = np.array([b1, b2], float)  # 2,
    
    # Se forem paralelas/singulares, retorne None
    try:
        # Verifica se o determinante é muito próximo de zero (matriz singular)
        det = np.linalg.det(Aeq)
        if abs(det) < EPS:
            return None
        
        # Resolve o sistema linear Aeq @ [x,y] = beq
        ponto = np.linalg.solve(Aeq, beq)
        return ponto
        
    except np.linalg.LinAlgError:
        # Matriz singular ou outro erro numérico
        return None


# Resolver PL 2D via pontos extremos 
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
    origem = np.array([0.0, 0.0])
    if np.all(A @ origem <= b + EPS):
        candidatos.append(origem)

    if not candidatos:
        raise ValueError("Nenhum ponto viável encontrado!")

    P = np.array(candidatos)
    valores = P @ c
    k = int(np.argmax(valores))
    return P[k], float(valores[k]), P, valores


# Teste do resolvedor
if __name__ == "__main__":
    print("=== RESOLVEDOR DE PROGRAMAÇÃO LINEAR 2D ===")
    print("\nProblema:")
    print("Maximizar z = 3x + 5y")
    print("Sujeito a:")
    print("  2x + y <= 100")
    print("  x + 2y <= 80")
    print("  x >= 0")
    print("  y >= 0")
    
    try:
        solucao_otima, valor_otimo, todos_pontos, todos_valores = resolve_lp_2d(A, b, c)
        
        print(f"\n=== RESULTADOS ===")
        print(f"Solução ótima: x = {solucao_otima[0]:.4f}, y = {solucao_otima[1]:.4f}")
        print(f"Valor ótimo: z = {valor_otimo:.4f}")
        
        print(f"\nTodos os pontos extremos encontrados:")
        for i, (ponto, valor) in enumerate(zip(todos_pontos, todos_valores)):
            print(f"  Ponto {i+1}: ({ponto[0]:.4f}, {ponto[1]:.4f}) -> z = {valor:.4f}")
            
        # Verificação da solução
        print(f"\n=== VERIFICAÇÃO ===")
        print(f"Verificando restrições na solução ótima:")
        restricoes_satisfeitas = A @ solucao_otima <= b + EPS
        for i, satisfeita in enumerate(restricoes_satisfeitas):
            valor_restricao = (A @ solucao_otima)[i]
            if i == 0:
                print(f"  2x + y = {valor_restricao:.4f} <= 100? {satisfeita}")
            elif i == 1:
                print(f"  x + 2y = {valor_restricao:.4f} <= 80? {satisfeita}")
            elif i == 2:
                print(f"  x = {solucao_otima[0]:.4f} >= 0? {satisfeita}")
            elif i == 3:
                print(f"  y = {solucao_otima[1]:.4f} >= 0? {satisfeita}")
        
        print(f"\nTodas as restrições satisfeitas: {np.all(restricoes_satisfeitas)}")
        
    except Exception as e:
        print(f"Erro ao resolver o problema: {e}")