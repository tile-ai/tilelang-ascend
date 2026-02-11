file(REMOVE_RECURSE
  "libtilelang.a"
  "libtilelang.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/tilelang_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
