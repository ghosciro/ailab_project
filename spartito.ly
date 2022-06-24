upper = \fixed a,,,  { 
\clef treble
\key c \major
\time 2/4

 << {gis''''''4}>> << {f'''''''1}>> r1

}
lower = \fixed a,,,  {
\clef bass
\key c \major
\time 2/4

 r1 r1 r8

}
\score {
\new PianoStaff \with { instrumentName = "Piano" }
<<
\new Staff = "upper" \upper
\new Staff = "lower" \lower
>>
\layout { }
\midi { }
}