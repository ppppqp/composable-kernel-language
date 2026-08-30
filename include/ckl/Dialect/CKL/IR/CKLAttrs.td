#ifndef CKL_DIALECT_CKL_IR_CKLATTRS_TD
#define CKL_DIALECT_CKL_IR_CKLATTRS_TD

include "CKLDialect.td"

// TODO: Add a more structured representation of these attributes, rather than just storing the string representation of the CKLCore object.
def CKL_SpaceAttr : CKL_Attr<"Space", "space"> {
  let summary = "A serialized CKL index space";
  let parameters = (ins StringRefParameter<"canonical CKLCore space spelling">:$value);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def CKL_IndexMapAttr : CKL_Attr<"IndexMap", "index_map"> {
  let summary = "A serialized CKL index map";
  let parameters = (ins StringRefParameter<"canonical CKLCore map spelling">:$value);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def CKL_DistributionAttr : CKL_Attr<"Distribution", "distribution"> {
  let summary = "A serialized CKL physical tile distribution";
  let parameters = (ins StringRefParameter<"canonical CKLCore distribution spelling">:$value);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif
