// expected-error @+2 {{invalid CKL index space}}
// expected-error @+1 {{failed to parse CKL_TileType parameter 'space'}}
func.func @invalid_space(%arg0: !ckl.tile<f32, #ckl.space<"not-a-space">>)
