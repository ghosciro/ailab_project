upper = \fixed a,,,  { 
\clef treble
\key c \major
\time 4/4

 << {a'''''8~4} {gis'''''8}>> r8 << {e'''''4}>> << {a'''''4}>> r1

}
lower = \fixed a,,,  {
\clef bass
\key c \major
\time 4/4

 << {fis''''8~8} {cis''''8}>> r8 << {d'''4}>> r8 << {e''''1}>> r8

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