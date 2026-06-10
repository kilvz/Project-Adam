class WorkingMemory:
    def __init__(self, max_turns=8):
        self.max_turns = max_turns
        self.turns = []

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_context(self, n=None):
        return self.turns[-(n or self.max_turns):]

    def clear(self):
        self.turns = []
