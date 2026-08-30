#include "ckl/Core/Layout/IndexMap.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ckl::core {

Axis::Axis(std::string name, std::int64_t extent) : name(std::move(name)), extent(extent) {}

bool Axis::operator==(const Axis &other) const {
  return name == other.name && extent == other.extent && extentSymbol == other.extentSymbol;
}

Axis Axis::symbolic(std::string name, std::string extentSymbol) {
  if (extentSymbol.empty())
    throw std::invalid_argument("symbolic extent name must not be empty");
  Axis axis(std::move(name), -1);
  axis.extentSymbol = std::move(extentSymbol);
  return axis;
}

namespace {
std::size_t combineHash(std::size_t seed, std::size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}
} // namespace

IndexSpace::IndexSpace(std::vector<Axis> axes) : axes_(std::move(axes)), profile_("[") {
  auto root = std::make_shared<Structure>();
  root->name = "flat";
  for (const Axis &axis : axes_) {
    if (axis.isStatic() && axis.extent <= 0)
      throw std::invalid_argument("index-space extents must be positive");
    if (profile_.size() > 1)
      profile_ += ',';
    profile_ += '*';
    auto leaf = std::make_shared<Structure>();
    leaf->name = axis.name;
    leaf->axis = axis;
    root->children.push_back(std::move(leaf));
  }
  profile_ += ']';
  structure_ = std::move(root);
}

IndexSpace::IndexSpace(std::vector<Axis> axes, std::string profile,
                       std::shared_ptr<const Structure> structure)
    : axes_(std::move(axes)), profile_(std::move(profile)), structure_(std::move(structure)) {}

IndexSpace IndexSpace::product(std::string name, std::vector<IndexSpace> children) {
  std::vector<Axis> axes;
  std::string profile = name + '(';
  auto root = std::make_shared<Structure>();
  root->name = name;
  for (std::size_t i = 0; i < children.size(); ++i) {
    if (i)
      profile += ',';
    profile += children[i].profile();
    axes.insert(axes.end(), children[i].axes().begin(), children[i].axes().end());
    root->children.push_back(children[i].structure());
  }
  profile += ')';
  return IndexSpace(std::move(axes), std::move(profile), std::move(root));
}

std::int64_t IndexSpace::volume() const {
  if (!isStatic())
    throw std::logic_error("symbolic index space must be instantiated before enumeration");
  std::int64_t result = 1;
  for (const Axis &axis : axes_)
    result *= axis.extent;
  return result;
}

bool IndexSpace::isStatic() const {
  return std::all_of(axes_.begin(), axes_.end(), [](const Axis &axis) { return axis.isStatic(); });
}

IndexSpace
IndexSpace::instantiate(const std::unordered_map<std::string, std::int64_t> &bindings) const {
  std::vector<Axis> axes;
  axes.reserve(axes_.size());
  for (const Axis &axis : axes_) {
    if (axis.isStatic()) {
      axes.push_back(axis);
      continue;
    }
    auto binding = bindings.find(*axis.extentSymbol);
    if (binding == bindings.end() || binding->second <= 0)
      throw std::invalid_argument("missing or non-positive symbolic extent binding: " +
                                  *axis.extentSymbol);
    axes.push_back({axis.name, binding->second});
  }
  return IndexSpace(std::move(axes));
}

bool IndexSpace::contains(const std::vector<std::int64_t> &point) const {
  if (point.size() != axes_.size())
    return false;
  for (std::size_t i = 0; i < point.size(); ++i) {
    if (point[i] < 0 || point[i] >= axes_[i].extent)
      return false;
  }
  return true;
}

bool IndexSpace::sameShape(const IndexSpace &other) const {
  if (rank() != other.rank())
    return false;
  for (std::size_t i = 0; i < rank(); ++i) {
    if (axes_[i].extent != other.axes_[i].extent ||
        axes_[i].extentSymbol != other.axes_[i].extentSymbol)
      return false;
  }
  return true;
}

std::size_t IndexSpace::hash() const {
  std::size_t result = std::hash<std::string>{}(profile_);
  for (const Axis &axis : axes_)
    result = combineHash(result, axis.isStatic() ? std::hash<std::int64_t>{}(axis.extent)
                                                 : std::hash<std::string>{}(*axis.extentSymbol));
  return result;
}

std::string IndexSpace::str() const {
  std::ostringstream os;
  os << '(';
  for (std::size_t i = 0; i < axes_.size(); ++i) {
    if (i)
      os << ", ";
    os << axes_[i].name << ':';
    if (axes_[i].isStatic())
      os << axes_[i].extent;
    else
      os << '$' << *axes_[i].extentSymbol;
  }
  return os.str() + ')';
}

std::string IndexSpace::serialize() const {
  std::ostringstream os;
  os << "s(";
  for (std::size_t i = 0; i < axes_.size(); ++i) {
    if (i)
      os << ',';
    os << axes_[i].name << ':';
    if (axes_[i].isStatic())
      os << axes_[i].extent;
    else
      os << '$' << *axes_[i].extentSymbol;
  }
  return os.str() + ')';
}

struct IndexExpr::Node {
  Kind kind;
  std::int64_t value;
  std::shared_ptr<const Node> lhs;
  std::shared_ptr<const Node> rhs;
};

IndexExpr::IndexExpr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}

static IndexExpr makeLeaf(IndexExpr::Kind kind, std::int64_t value) {
  return kind == IndexExpr::Kind::Input ? IndexExpr::input(static_cast<std::size_t>(value))
                                        : IndexExpr::constant(value);
}

IndexExpr IndexExpr::input(std::size_t index) {
  return IndexExpr(
      std::make_shared<Node>(Node{Kind::Input, static_cast<std::int64_t>(index), {}, {}}));
}

IndexExpr IndexExpr::constant(std::int64_t value) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Constant, value, {}, {}}));
}

IndexExpr IndexExpr::add(IndexExpr lhs, IndexExpr rhs) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Add, 0, lhs.node_, rhs.node_}));
}

IndexExpr IndexExpr::multiply(IndexExpr lhs, IndexExpr rhs) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Multiply, 0, lhs.node_, rhs.node_}));
}

IndexExpr IndexExpr::floorDiv(IndexExpr value, std::int64_t divisor) {
  if (divisor <= 0)
    throw std::invalid_argument("floor-divisor must be positive");
  return IndexExpr(std::make_shared<Node>(Node{Kind::FloorDiv, divisor, value.node_, {}}));
}

IndexExpr IndexExpr::modulo(IndexExpr value, std::int64_t modulus) {
  if (modulus <= 0)
    throw std::invalid_argument("modulus must be positive");
  return IndexExpr(std::make_shared<Node>(Node{Kind::Modulo, modulus, value.node_, {}}));
}

IndexExpr IndexExpr::bitXor(IndexExpr lhs, IndexExpr rhs) {
  return IndexExpr(std::make_shared<Node>(Node{Kind::Xor, 0, lhs.node_, rhs.node_}));
}

std::int64_t IndexExpr::evaluate(const std::vector<std::int64_t> &inputs) const {
  std::function<std::int64_t(const std::shared_ptr<const Node> &)> eval =
      [&](const std::shared_ptr<const Node> &node) -> std::int64_t {
    switch (node->kind) {
    case Kind::Input:
      if (node->value < 0 || static_cast<std::size_t>(node->value) >= inputs.size())
        throw std::out_of_range("index expression input is out of range");
      return inputs[static_cast<std::size_t>(node->value)];
    case Kind::Constant:
      return node->value;
    case Kind::Add:
      return eval(node->lhs) + eval(node->rhs);
    case Kind::Multiply:
      return eval(node->lhs) * eval(node->rhs);
    case Kind::FloorDiv:
      return eval(node->lhs) / node->value;
    case Kind::Modulo:
      return eval(node->lhs) % node->value;
    case Kind::Xor:
      return eval(node->lhs) ^ eval(node->rhs);
    }
    throw std::logic_error("unknown index expression kind");
  };
  return eval(node_);
}

IndexExpr IndexExpr::substitute(const std::vector<IndexExpr> &replacements) const {
  std::function<IndexExpr(const std::shared_ptr<const Node> &)> rewrite =
      [&](const std::shared_ptr<const Node> &node) -> IndexExpr {
    switch (node->kind) {
    case Kind::Input:
      if (node->value < 0 || static_cast<std::size_t>(node->value) >= replacements.size())
        throw std::out_of_range("substitution input is out of range");
      return replacements[static_cast<std::size_t>(node->value)];
    case Kind::Constant:
      return makeLeaf(node->kind, node->value);
    case Kind::Add:
      return add(rewrite(node->lhs), rewrite(node->rhs));
    case Kind::Multiply:
      return multiply(rewrite(node->lhs), rewrite(node->rhs));
    case Kind::FloorDiv:
      return floorDiv(rewrite(node->lhs), node->value);
    case Kind::Modulo:
      return modulo(rewrite(node->lhs), node->value);
    case Kind::Xor:
      return bitXor(rewrite(node->lhs), rewrite(node->rhs));
    }
    throw std::logic_error("unknown index expression kind");
  };
  return rewrite(node_);
}

IndexExpr IndexExpr::normalize() const {
  // TODO: constant propagation and other simplifications
  switch (node_->kind) {
  case Kind::Input:
    return input(static_cast<std::size_t>(node_->value));
  case Kind::Constant:
    return constant(node_->value);
  case Kind::FloorDiv:
    return floorDiv(IndexExpr(node_->lhs).normalize(), node_->value);
  case Kind::Modulo:
    return modulo(IndexExpr(node_->lhs).normalize(), node_->value);
  case Kind::Add:
  case Kind::Multiply:
  case Kind::Xor: {
    // commutative operators are normalized by sorting the string representations of their operands
    IndexExpr lhs = IndexExpr(node_->lhs).normalize();
    IndexExpr rhs = IndexExpr(node_->rhs).normalize();
    if (rhs.str() < lhs.str())
      std::swap(lhs, rhs);
    if (node_->kind == Kind::Add)
      return add(lhs, rhs);
    if (node_->kind == Kind::Multiply)
      return multiply(lhs, rhs);
    return bitXor(lhs, rhs);
  }
  }
  throw std::logic_error("unknown index expression kind");
}

std::size_t IndexExpr::hash() const { return std::hash<std::string>{}(normalize().str()); }

/*
  i(0)             input dimension 0
  c(3)             constant 3
  a(i(0),c(1))     d0 + 1
  m(i(0),c(4))     d0 * 4
  d(i(0),c...)     floor division
  r(i(0),...)      modulo
  x(i(0),c(3))     XOR swizzle
*/
std::string IndexExpr::serialize() const {
  switch (node_->kind) {
  case Kind::Input:
    return "i(" + std::to_string(node_->value) + ')';
  case Kind::Constant:
    return "c(" + std::to_string(node_->value) + ')';
  case Kind::Add:
    return "a(" + IndexExpr(node_->lhs).serialize() + ',' + IndexExpr(node_->rhs).serialize() + ')';
  case Kind::Multiply:
    return "m(" + IndexExpr(node_->lhs).serialize() + ',' + IndexExpr(node_->rhs).serialize() + ')';
  case Kind::FloorDiv:
    return "d(" + IndexExpr(node_->lhs).serialize() + ',' + std::to_string(node_->value) + ')';
  case Kind::Modulo:
    return "r(" + IndexExpr(node_->lhs).serialize() + ',' + std::to_string(node_->value) + ')';
  case Kind::Xor:
    return "x(" + IndexExpr(node_->lhs).serialize() + ',' + IndexExpr(node_->rhs).serialize() + ')';
  }
  throw std::logic_error("unknown index expression kind");
}

std::string IndexExpr::str() const {
  std::function<std::string(const std::shared_ptr<const Node> &)> print =
      [&](const std::shared_ptr<const Node> &node) -> std::string {
    if (node->kind == Kind::Input)
      return "d" + std::to_string(node->value);
    if (node->kind == Kind::Constant)
      return std::to_string(node->value);
    if (node->kind == Kind::FloorDiv)
      return "floordiv(" + print(node->lhs) + ", " + std::to_string(node->value) + ')';
    if (node->kind == Kind::Modulo)
      return "mod(" + print(node->lhs) + ", " + std::to_string(node->value) + ')';
    const char *op = node->kind == Kind::Add        ? " + "
                     : node->kind == Kind::Multiply ? " * "
                                                    : " xor ";
    return '(' + print(node->lhs) + op + print(node->rhs) + ')';
  };
  return print(node_);
}

struct IndexPredicate::Node {
  enum class Kind { Always, Compare, And } kind = Kind::Always;
  Comparison comparison = Comparison::Equal;
  std::optional<IndexExpr> lhs;
  std::optional<IndexExpr> rhs;
  std::shared_ptr<const Node> leftPredicate;
  std::shared_ptr<const Node> rightPredicate;
};

IndexPredicate::IndexPredicate(std::shared_ptr<const Node> node) : node_(std::move(node)) {}

IndexPredicate IndexPredicate::always() { return IndexPredicate(std::make_shared<Node>(Node{})); }

IndexPredicate IndexPredicate::compare(IndexExpr lhs, Comparison comparison, IndexExpr rhs) {
  Node node{};
  node.kind = Node::Kind::Compare;
  node.comparison = comparison;
  node.lhs = std::move(lhs);
  node.rhs = std::move(rhs);
  return IndexPredicate(std::make_shared<Node>(std::move(node)));
}

IndexPredicate IndexPredicate::logicalAnd(IndexPredicate lhs, IndexPredicate rhs) {
  Node node{};
  node.kind = Node::Kind::And;
  node.leftPredicate = lhs.node_;
  node.rightPredicate = rhs.node_;
  return IndexPredicate(std::make_shared<Node>(std::move(node)));
}

bool IndexPredicate::evaluate(const std::vector<std::int64_t> &inputs) const {
  std::function<bool(const std::shared_ptr<const Node> &)> eval =
      [&](const std::shared_ptr<const Node> &node) {
        if (node->kind == Node::Kind::Always)
          return true;
        if (node->kind == Node::Kind::And)
          return eval(node->leftPredicate) && eval(node->rightPredicate);
        const auto lhs = node->lhs->evaluate(inputs);
        const auto rhs = node->rhs->evaluate(inputs);
        switch (node->comparison) {
        case Comparison::Less:
          return lhs < rhs;
        case Comparison::LessEqual:
          return lhs <= rhs;
        case Comparison::Equal:
          return lhs == rhs;
        case Comparison::NotEqual:
          return lhs != rhs;
        case Comparison::GreaterEqual:
          return lhs >= rhs;
        case Comparison::Greater:
          return lhs > rhs;
        }
        return false;
      };
  return eval(node_);
}

IndexPredicate IndexPredicate::substitute(const std::vector<IndexExpr> &replacements) const {
  std::function<IndexPredicate(const std::shared_ptr<const Node> &)> rewrite =
      [&](const std::shared_ptr<const Node> &node) {
        if (node->kind == Node::Kind::Always)
          return always();
        if (node->kind == Node::Kind::And)
          return logicalAnd(rewrite(node->leftPredicate), rewrite(node->rightPredicate));
        return compare(node->lhs->substitute(replacements), node->comparison,
                       node->rhs->substitute(replacements));
      };
  return rewrite(node_);
}

std::string IndexPredicate::str() const {
  std::function<std::string(const std::shared_ptr<const Node> &)> print =
      [&](const std::shared_ptr<const Node> &node) -> std::string {
    if (node->kind == Node::Kind::Always)
      return "true";
    if (node->kind == Node::Kind::And)
      return '(' + print(node->leftPredicate) + " and " + print(node->rightPredicate) + ')';
    static const char *symbols[] = {"<", "<=", "==", "!=", ">=", ">"};
    return '(' + node->lhs->str() + ' ' + symbols[static_cast<int>(node->comparison)] + ' ' +
           node->rhs->str() + ')';
  };
  return print(node_);
}

std::size_t IndexPredicate::hash() const { return std::hash<std::string>{}(str()); }

std::string IndexPredicate::serialize() const {
  if (node_->kind == Node::Kind::Always)
    return "t";
  if (node_->kind == Node::Kind::And)
    return "and(" + IndexPredicate(node_->leftPredicate).serialize() + ',' +
           IndexPredicate(node_->rightPredicate).serialize() + ')';
  return "cmp(" + std::to_string(static_cast<int>(node_->comparison)) + ',' +
         node_->lhs->serialize() + ',' + node_->rhs->serialize() + ')';
}

IndexMap::IndexMap(IndexSpace domain, IndexSpace codomain, std::vector<IndexExpr> results,
                   IndexPredicate predicate)
    : domain_(std::move(domain)), codomain_(std::move(codomain)), results_(std::move(results)),
      predicate_(std::move(predicate)) {
  if (results_.size() != codomain_.rank())
    throw std::invalid_argument("index map must produce one expression per codomain axis");
}

IndexMap IndexMap::identity(IndexSpace space) {
  std::vector<IndexExpr> results;
  for (std::size_t i = 0; i < space.rank(); ++i)
    results.push_back(IndexExpr::input(i));
  return IndexMap(space, space, std::move(results));
}

IndexMap IndexMap::permutation(IndexSpace domain, std::vector<std::size_t> order,
                               std::vector<std::string> resultNames) {
  if (order.size() != domain.rank())
    throw std::invalid_argument("permutation rank mismatch");
  std::vector<bool> seen(order.size());
  std::vector<Axis> axes;
  std::vector<IndexExpr> results;
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] >= order.size() || seen[order[i]])
      throw std::invalid_argument("invalid permutation");
    seen[order[i]] = true;
    Axis axis = domain.axes()[order[i]];
    axis.name = resultNames.empty() ? axis.name : resultNames[i];
    axes.push_back(std::move(axis));
    results.push_back(IndexExpr::input(order[i]));
  }
  return IndexMap(std::move(domain), IndexSpace(std::move(axes)), std::move(results));
}

static IndexExpr linearize(const IndexSpace &space) {
  IndexExpr result = IndexExpr::constant(0);
  std::int64_t stride = 1;
  for (std::size_t i = space.rank(); i-- > 0;) {
    result = IndexExpr::add(result,
                            IndexExpr::multiply(IndexExpr::input(i), IndexExpr::constant(stride)));
    stride *= space.axes()[i].extent;
  }
  return result;
}

IndexMap IndexMap::reshape(IndexSpace domain, IndexSpace codomain) {
  if (domain.volume() != codomain.volume())
    throw std::invalid_argument("reshape spaces must have equal volume");
  IndexExpr linear = linearize(domain);
  std::vector<IndexExpr> results;
  std::int64_t divisor = codomain.volume();
  for (const Axis &axis : codomain.axes()) {
    divisor /= axis.extent;
    results.push_back(IndexExpr::modulo(IndexExpr::floorDiv(linear, divisor), axis.extent));
  }
  return IndexMap(std::move(domain), std::move(codomain), std::move(results));
}

IndexMap IndexMap::strided(IndexSpace domain, std::vector<std::int64_t> strides,
                           std::int64_t offset) {
  if (strides.size() != domain.rank())
    throw std::invalid_argument("stride rank mismatch");
  IndexExpr result = IndexExpr::constant(offset);
  std::int64_t minAddress = offset;
  std::int64_t maxAddress = offset;
  for (std::size_t i = 0; i < strides.size(); ++i) {
    result = IndexExpr::add(
        result, IndexExpr::multiply(IndexExpr::input(i), IndexExpr::constant(strides[i])));
    const std::int64_t span = (domain.axes()[i].extent - 1) * strides[i];
    minAddress += std::min<std::int64_t>(0, span);
    maxAddress += std::max<std::int64_t>(0, span);
  }
  if (minAddress < 0)
    throw std::invalid_argument("strided layout produces negative addresses");
  return IndexMap(std::move(domain), IndexSpace({{"address", maxAddress + 1}}), {result});
}

std::vector<std::int64_t> IndexMap::apply(const std::vector<std::int64_t> &point) const {
  if (!domain_.contains(point))
    throw std::out_of_range("point is outside index-map domain");
  if (!predicate_.evaluate(point))
    throw std::out_of_range("point is outside index-map predicate");
  std::vector<std::int64_t> result;
  for (const IndexExpr &expr : results_)
    result.push_back(expr.evaluate(point));
  if (!codomain_.contains(result))
    throw std::out_of_range("index map produced a point outside its codomain");
  return result;
}

std::optional<std::vector<std::int64_t>>
IndexMap::tryApply(const std::vector<std::int64_t> &point) const {
  if (!domain_.contains(point) || !predicate_.evaluate(point))
    return std::nullopt;
  return apply(point);
}

IndexMap IndexMap::normalize() const {
  std::vector<IndexExpr> normalized;
  for (const auto &result : results_)
    normalized.push_back(result.normalize());
  return IndexMap(domain_, codomain_, std::move(normalized), predicate_);
}

std::size_t IndexMap::hash() const {
  std::size_t result = combineHash(domain_.hash(), codomain_.hash());
  for (const auto &expr : results_)
    result = combineHash(result, expr.hash());
  return combineHash(result, predicate_.hash());
}

std::string IndexMap::str() const {
  std::ostringstream os;
  os << domain_.str() << " -> " << codomain_.str() << " {";
  for (std::size_t i = 0; i < results_.size(); ++i) {
    if (i)
      os << ", ";
    os << results_[i].str();
  }
  return os.str() + '}';
}

namespace {
class MapParser {
public:
  explicit MapParser(const std::string &text) : text(text) {}

  void expect(char value) {
    if (position >= text.size() || text[position++] != value)
      throw std::invalid_argument("invalid serialized index map");
  }
  void expect(const std::string &value) {
    if (text.compare(position, value.size(), value) != 0)
      throw std::invalid_argument("invalid serialized index map");
    position += value.size();
  }
  std::int64_t integer() {
    std::size_t begin = position;
    if (position < text.size() && text[position] == '-')
      ++position;
    while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position])))
      ++position;
    if (begin == position || (text[begin] == '-' && begin + 1 == position))
      throw std::invalid_argument("expected integer in serialized index map");
    return std::stoll(text.substr(begin, position - begin));
  }
  std::string identifier() {
    std::size_t begin = position;
    while (position < text.size() && text[position] != ':' && text[position] != ',' &&
           text[position] != ')')
      ++position;
    if (begin == position)
      throw std::invalid_argument("expected axis name in serialized index map");
    return text.substr(begin, position - begin);
  }
  IndexSpace space() {
    expect("s(");
    std::vector<Axis> axes;
    if (position < text.size() && text[position] != ')') {
      while (true) {
        std::string name = identifier();
        expect(':');
        if (text.at(position) == '$') {
          ++position;
          axes.push_back(Axis::symbolic(std::move(name), identifier()));
        } else {
          axes.push_back({std::move(name), integer()});
        }
        if (text[position] != ',')
          break;
        ++position;
      }
    }
    expect(')');
    return IndexSpace(std::move(axes));
  }
  IndexExpr expression() {
    const char kind = text.at(position++);
    expect('(');
    if (kind == 'i' || kind == 'c') {
      auto value = integer();
      expect(')');
      return kind == 'i' ? IndexExpr::input(static_cast<std::size_t>(value))
                         : IndexExpr::constant(value);
    }
    IndexExpr lhs = expression();
    expect(',');
    if (kind == 'd' || kind == 'r') {
      auto value = integer();
      expect(')');
      return kind == 'd' ? IndexExpr::floorDiv(lhs, value) : IndexExpr::modulo(lhs, value);
    }
    IndexExpr rhs = expression();
    expect(')');
    if (kind == 'a')
      return IndexExpr::add(lhs, rhs);
    if (kind == 'm')
      return IndexExpr::multiply(lhs, rhs);
    if (kind == 'x')
      return IndexExpr::bitXor(lhs, rhs);
    throw std::invalid_argument("unknown expression in serialized index map");
  }
  IndexPredicate predicate() {
    if (text.at(position) == 't') {
      ++position;
      return IndexPredicate::always();
    }
    if (text.compare(position, 4, "and(") == 0) {
      position += 4;
      auto lhs = predicate();
      expect(',');
      auto rhs = predicate();
      expect(')');
      return IndexPredicate::logicalAnd(lhs, rhs);
    }
    expect("cmp(");
    auto comparison = static_cast<IndexPredicate::Comparison>(integer());
    expect(',');
    auto lhs = expression();
    expect(',');
    auto rhs = expression();
    expect(')');
    return IndexPredicate::compare(lhs, comparison, rhs);
  }
  bool done() const { return position == text.size(); }

private:
  const std::string &text;
  std::size_t position = 0;
};

std::string serializeSpace(const IndexSpace &space) { return space.serialize(); }
} // namespace

std::string IndexMap::serialize() const {
  std::ostringstream os;
  os << "map(" << serializeSpace(domain_) << ';' << serializeSpace(codomain_) << ";[";
  for (std::size_t i = 0; i < results_.size(); ++i) {
    if (i)
      os << ',';
    os << results_[i].serialize();
  }
  return os.str() + "];" + predicate_.serialize() + ')';
}

IndexSpace IndexSpace::deserialize(const std::string &text) {
  MapParser parser(text);
  IndexSpace space = parser.space();
  if (!parser.done())
    throw std::invalid_argument("trailing text in serialized index space");
  return space;
}

IndexMap IndexMap::deserialize(const std::string &text) {
  MapParser parser(text);
  parser.expect("map(");
  IndexSpace domain = parser.space();
  parser.expect(';');
  IndexSpace codomain = parser.space();
  parser.expect(';');
  parser.expect('[');
  std::vector<IndexExpr> results;
  if (codomain.rank() != 0) {
    for (std::size_t i = 0; i < codomain.rank(); ++i) {
      if (i)
        parser.expect(',');
      results.push_back(parser.expression());
    }
  }
  parser.expect(']');
  parser.expect(';');
  IndexPredicate predicate = parser.predicate();
  parser.expect(')');
  if (!parser.done())
    throw std::invalid_argument("trailing text in serialized index map");
  return IndexMap(std::move(domain), std::move(codomain), std::move(results), std::move(predicate));
}

std::string IndexMap::diagram(std::int64_t pointLimit) const {
  std::ostringstream os;
  os << domain_.str() << " -> " << codomain_.str() << '\n';
  if (domain_.volume() > pointLimit) {
    os << "<diagram omitted: " << domain_.volume() << " points exceeds limit>\n";
    return os.str();
  }
  for (const auto &point : enumerate(domain_)) {
    os << '[';
    for (std::size_t i = 0; i < point.size(); ++i) {
      if (i)
        os << ',';
      os << point[i];
    }
    os << "] -> ";
    auto result = tryApply(point);
    if (!result) {
      os << "inactive\n";
      continue;
    }
    os << '[';
    for (std::size_t i = 0; i < result->size(); ++i) {
      if (i)
        os << ',';
      os << (*result)[i];
    }
    os << "]\n";
  }
  return os.str();
}

// NOTE: This assumes a canonical row-major correspondence between factorizations of the same
// volume. A more general implementation must preserve nested factor structure and prove that a
// refinement is compatible.
IndexMap compose(const IndexMap &outer, const IndexMap &inner) {
  if (inner.codomain().volume() != outer.domain().volume())
    throw std::invalid_argument("composition intermediate spaces have different volumes");
  std::vector<IndexExpr> replacements = inner.results();
  if (!inner.codomain().sameShape(outer.domain())) {
    auto refinement = computeMutualRefinement(inner.codomain(), outer.domain());
    if (!refinement)
      throw std::invalid_argument("composition has no common factor refinement");
    IndexMap bridge =
        compose(IndexMap::reshape(refinement->commonSpace, outer.domain()), refinement->fromLeft);
    replacements.clear();
    for (const IndexExpr &expr : bridge.results())
      replacements.push_back(expr.substitute(inner.results()));
  }
  std::vector<IndexExpr> results;
  for (const IndexExpr &expr : outer.results())
    // for each axis in the outer map, substitute the inner map's expressions for its inputs
    results.push_back(expr.substitute(replacements));
  IndexPredicate predicate =
      IndexPredicate::logicalAnd(inner.predicate(), outer.predicate().substitute(replacements));
  return IndexMap(inner.domain(), outer.codomain(), std::move(results), std::move(predicate));
}

// TODO: this is a canonical prime-factor refinement. It's still simpler than the full
// pullback/pushforward described by layout-categories here
// https://arxiv.org/pdf/2601.05972
std::optional<FactorRefinement> computeMutualRefinement(const IndexSpace &left,
                                                        const IndexSpace &right) {
  if (left.volume() != right.volume())
    return std::nullopt;
  std::int64_t remaining = left.volume();
  std::vector<std::int64_t> factors;
  for (std::int64_t divisor = 2; divisor * divisor <= remaining; ++divisor) {
    while (remaining % divisor == 0) {
      factors.push_back(divisor);
      remaining /= divisor;
    }
  }
  if (remaining > 1)
    factors.push_back(remaining);
  if (factors.empty())
    factors.push_back(1);
  std::vector<Axis> axes;
  for (std::size_t i = 0; i < factors.size(); ++i)
    axes.push_back({"f" + std::to_string(i), factors[i]});
  IndexSpace common = IndexSpace::product("refinement", {IndexSpace(std::move(axes))});
  IndexMap fromLeft = IndexMap::reshape(left, common);
  IndexMap fromRight = IndexMap::reshape(right, common);
  return FactorRefinement{common, std::move(fromLeft), std::move(fromRight), std::move(factors)};
}

std::vector<std::vector<std::int64_t>> enumerate(const IndexSpace &space) {
  std::vector<std::vector<std::int64_t>> points;
  std::vector<std::int64_t> point(space.rank());
  std::function<void(std::size_t)> visit = [&](std::size_t axis) {
    if (axis == space.rank()) {
      points.push_back(point);
      return;
    }
    for (std::int64_t i = 0; i < space.axes()[axis].extent; ++i) {
      point[axis] = i;
      visit(axis + 1);
    }
  };
  visit(0);
  return points;
}

// NOTE: currently the implementation enumerates every point and compare the results
// Not scalable. Move to a more efficient symbolic proof in the future (Presburger reasoning)
EquivalenceResult proveEquivalent(const IndexMap &lhs, const IndexMap &rhs,
                                  std::int64_t exhaustiveLimit) {
  if (!lhs.domain().sameShape(rhs.domain()) || !lhs.codomain().sameShape(rhs.codomain()))
    return {EquivalenceResult::Status::NotEquivalent, std::nullopt,
            "source or result shapes differ"};
  if (lhs.domain().volume() > exhaustiveLimit)
    return {EquivalenceResult::Status::Unknown, std::nullopt,
            "domain exceeds exhaustive proof limit"};
  for (const auto &point : enumerate(lhs.domain())) {
    auto lhsValue = lhs.tryApply(point);
    auto rhsValue = rhs.tryApply(point);
    if (lhsValue != rhsValue)
      return {EquivalenceResult::Status::NotEquivalent, point, "maps differ at witness point"};
  }
  return {EquivalenceResult::Status::Equivalent, std::nullopt, "exhaustive finite-domain proof"};
}

} // namespace ckl::core
