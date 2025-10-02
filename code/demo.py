def prob2(self, y:Wordtype, z: Wordtype) -> float:
        return ((self.event_count[y,z] + self.lambda_ * self.vocab_size * prob1(z)) / 
                (self.context_count[y] + self.lambda_ * self.vocab_size))
    def prob1(self, z: Wordtype) -> float:
        return ((self.event_count[z] + self.lambda_) / 
                (self.context_count[z] + self.lambda_ * self.vocab_size))
    
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # TODO: Reimplement me so that I do backoff
        return ((self.event_count[x,y,z] + self.lambda_ * self.vocab_size * prob2(y,z)) / 
                (self.context_count[x,y] + self.lambda_ * self.vocab_size))