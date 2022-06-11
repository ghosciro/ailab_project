upper = \fixed a,,,  { 
\clef treble
\key c \major
\time 4/4

 r1 r2

}
lower = \fixed a,,,  {
\clef bass
\key c \major
\time 4/4

 << {fis''1}>> r2

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