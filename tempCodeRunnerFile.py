   def manage(self, trash_list):
        for t in trash_list:
            if self.model.space.positions[self] == self.model.space.positions[t]:
                t.collected = True
                self.trash_collected += 1