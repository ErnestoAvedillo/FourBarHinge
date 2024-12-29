class MuelleTraccion:
    def __init__(self, longitud_inicial:float, constante:float):
        self.longitud_inicial = longitud_inicial
        self.constante = constante
        
    def get_force (self, longitud_actual:float):
        if longitud_actual < self.longitud_inicial:
            return 0
        return (longitud_actual - self.longitud_inicial) * self.constante
