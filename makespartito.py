import mingus.extra.lilypond as LilyPond
import mingus.containers as co

b =co.Track()
b + "C"
b + "E"
b + "G"
b + "B"
b + "A"
barr = LilyPond.from_Track(b)
print(barr)
LilyPond.to_png(barr, "my_first_bar")