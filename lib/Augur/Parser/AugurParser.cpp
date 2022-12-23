#include "augur/Parser/AugurParser.h"
#include "mlir/Support/LogicalResult.h"
#include <iostream>
#include <optional>
#include <unordered_map>

using namespace mlir;
using namespace aug;

// sig  ::=                       % Empty signature
//        | decl sig              % Constant declaration

// decl ::= id : term.            % a : K  or  c : A
//        | defn.                 % definition, usually d : A = M
//        | %abbrev adecl.        % abbreviation
//        | %infix ixdecl.        % operator declaration
//        | %prefix pxdecl.       % operator declaration
//        | %postfix pxdecl.      % operator declaration
//        | %name namepref.       % name preference declaration
//        | %query qdecl.         % query declaration
//        | %clause defn.         % clause definition
//        | sdecl.                % solve declaration
//        | %tabled id.           % table declaration
//        | %querytabled qtdecl   % tabled query declaration
//        | %deterministic ddecl. % deterministic declaration
//        | %mode mdecl.          % mode declaration
//        | %terminates tdecl.    % termination declaration
//        | %reduces rdecl.       % reduction declaration
//        | %block id : bdecl.    % block declaration
//        | %worlds wdecl.        % worlds declaration
//        | %total tdecl.         % totality declaration
//        | %freeze ids.          % freeze declaration
//        | %theorem thdecl.      % theorem declaration
//        | %prove pdecl.         % prove declaration
//        | %establish pdecl.     % prove declaration, do not use as lemma later
//        | %assert callpats.     % assert theorem (requires Twelf.unsafe)
//        | %use domain.          % installs constraint domain

// defn ::= id : term = term      % d : A = M or d : K = A
//        | id = term             % d = M or d = A
//        | _ : term = term       % anonymous definition, for type-checking
//        | _ = term              % anonymous definition, for type-checking

// sdecl ::= %define binding sdecl % term binding
//         | %solve id : term      % solve with proof term
//         | %solve _ : term       % solve without proof term

// ids   ::=                      % empty sequence
//         | id ids               % identifier follwed by more

// term ::= type                  % type
//        | id                    % variable x or constant a or c
//        | term -> term          % A -> B
//        | term <- term          % A <- B, same as B -> A
//        | {id : term} term      % Pi x:A. K  or  Pi x:A. B
//        | [id : term] term      % lambda x:A. B  or  lambda x:A. M
//        | term term             % A M  or  M N
//        | term : term           % explicit type ascription
//        | _                     % hole, to be filled by term reconstruction
//        | {id} term             % same as {id:_} term
//        | [id] term             % same as [id:_] term

/// This class implements `Optional` functionality for ParseResult. We don't
/// directly use Optional here, because it provides an implicit conversion
/// to 'bool' which we want to avoid. This class is used to implement tri-state
/// 'parseOptional' functions that may have a failure mode when parsing that
/// shouldn't be attributed to "not present".
namespace {
class OptionalParseResult {
public:
  OptionalParseResult() = default;
  OptionalParseResult(LogicalResult result) : impl(result) {}
  OptionalParseResult(ParseResult result) : impl(result) {}
  OptionalParseResult(std::nullopt_t) : impl(std::nullopt) {}

  /// Returns true if we contain a valid ParseResult value.
  bool has_value() const { return impl.has_value(); }

  /// Access the internal ParseResult value.
  ParseResult value() const { return *impl; }
  ParseResult operator*() const { return value(); }

private:
  Optional<ParseResult> impl;
};
} // namespace

namespace {
struct Bindings {
  struct Scope {
    Scope(Bindings &bindings) : bindings(bindings) {
      outerScope = bindings.currentScope;
      bindings.currentScope = this;
    }
    ~Scope() {
      for (auto &[name, value] : oldValues)
        bindings.map[name] = value;
      bindings.currentScope = outerScope;
    }
    Bindings &bindings;
    Scope *outerScope;
    std::vector<std::pair<std::string, Object *>> oldValues;
  };

  void insert(std::string name, Object *value) {
    auto &entry = map[name];
    currentScope->oldValues.emplace_back(name, entry);
    entry = value;
  }

  Object *lookup(std::string name) { return map[name]; }

private:
  Scope *currentScope = nullptr;
  std::unordered_map<std::string, Object *> map;
};
} // namespace

namespace {
struct Parser {
  Parser(VirtualMachine &vm, std::string_view code) : vm(vm), code(code) {}

  bool isSpace(unsigned char ch) {
    return ch == ' ' || ch == '\r' || ch == '\n' || ch == '\f' || ch == '\t' ||
           ch == '\v';
  }

  void finishLine() {
    while (!code.empty()) {
      auto ch = code.front();
      code.remove_prefix(1);
      if (ch == '\r' || ch == '\n')
        break;
    }
  }

  ParseResult skipWhitespace() {
    while (!code.empty()) {
      auto ch = code.front();
      if (ch == '#')
        finishLine();
      else if (isSpace(code.front()))
        code.remove_prefix(1);
      else
        break;
    }

    return success();
  }

  LogicalResult parseOptionalChar(unsigned char ch) {
    std::cout << "! parseOptionalChar begin " << ch << "\n";
    if (code.empty() || code.front() != ch)
      return failure();
    code.remove_prefix(1);
    std::cout << "! parseOptionalChar end\n";
    return success();
  }

  ParseResult parseChar(unsigned char ch) {
    if (failed(parseOptionalChar(ch))) {
      std::cout << "expected char " << ch << std::endl;
      return failure();
    }
    return success();
  }

  LogicalResult parseOptionalString(std::string_view string) {
    std::cout << "! parseOptionalString begin " << string << "\n";
    auto size = string.size();
    if (code.size() < size)
      return failure();
    for (std::size_t i = 0; i < size; ++i) {
      if (code[i] != string[i])
        return failure();
    }
    code.remove_prefix(size);
    std::cout << "! parseOptionalString end\n";
    return success();
  }

  bool isLeadingID(unsigned char ch) {
    return ('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z');
  }

  bool isID(unsigned char ch) {
    return isLeadingID(ch) || (ch == '-') || ('0' <= ch && ch <= '9');
  }

  LogicalResult parseOptionalID(std::string &result) {
    std::cout << "! parseOptionalID begin\n";
    std::cout << "remainder:\n[[[\n" << code << "\n]]]\n";
    auto size = code.size();

    if (size == 0 || !isLeadingID(code[0]))
      return failure();
    std::size_t count = 1;
    while (count < size && isID(code[count]))
      count++;
    result = code.substr(0, count);
    std::cout << "parsed an id: " << result << "\n";
    std::cout << "! parseOptionalID end\n";
    code.remove_prefix(count);
    return success();
  }

  ParseResult parseID(std::string &result) {
    std::cout << "! parseID begin\n";
    if (failed(parseOptionalID(result))) {
      std::cout << "expected ID\n";
      return failure();
    }
    std::cout << "! parseID end\n";
    return success();
  }

  /// assign ::= = term
  // ParseResult parseAssignation() { if (parseChar('=')) }

  ParseResult parseDefn() { return success(); }

  // decl ::= id : term.            % a : K  or  c : A
  //        | defn.                 % definition, usually d : A = M
  ParseResult parseDecl(Object *&decl) {
    std::cout << "! parseDecl begin\n";
    skipWhitespace();

    std::string id;
    if (parseID(id))
      return failure();
    std::cout << " id: " << id << "\n";
    auto atom = vm.create<Atom>(id);
    bindings.insert(id, atom);

    skipWhitespace();

    // TODO: the type annotation is not strictly neccessary.
    Object *type;
    if (parseChar(':') || skipWhitespace() || parseTerm(type) ||
        skipWhitespace())
      return failure();

    if (succeeded(parseOptionalChar('='))) {
      std::cout << "! parsing def\n";
      skipWhitespace();
      Object *value;
      if (parseTerm(value))
        return failure();
      decl = vm.create<Defn>(atom, type, value);
    } else {
      decl = vm.create<Decl>(atom, type);
    }

    if (parseChar('.'))
      return failure();

    std::cout << "! parseDecl end " << *decl << "\n";

    return success();
  }

  OptionalParseResult parseOptionalSubterm(Object *&term) {
    std::cout << "! parseOptionalSubterm begin\n";
    if (succeeded(parseOptionalChar('('))) {
      skipWhitespace();
      if (parseTerm(term))
        return failure();
      skipWhitespace();
      if (parseChar(')'))
        return failure();
      return success();
    }

    std::string id;
    if (succeeded(parseOptionalID(id))) {
      term = bindings.lookup(id);
      if (!term) {
        std::cout << "id \"" << id << "\" not found\n ";
        return failure();
      }
      return success();
    }

    std::cout << "! parseOptionalSubterm end\n";
    return std::nullopt;
  }

  OptionalParseResult parseOptionalTerm(Object *&term) {
    std::cout << "! parseOptionalTerm begin\n";

    if (succeeded(parseOptionalChar('{'))) {
      std::cout << "! parsePi begin\n";
      std::string variable;
      Object *type;
      if (parseID(variable) || skipWhitespace() || parseChar(':') ||
          skipWhitespace() || parseTerm(type) || skipWhitespace() ||
          parseChar('}') || skipWhitespace())
        return failure();
      auto atom = vm.create<Atom>(variable);

      // Add the binding.
      Bindings::Scope scope(bindings);
      bindings.insert(variable, atom);

      // Parse the body of the Pi.
      Object *body;
      if (parseTerm(body))
        return failure();

      term = vm.create<Pi>(atom, type, body);
      std::cout << "! parsePi end\n";

      return success();
    } else if (succeeded(parseOptionalChar('['))) {
      std::cout << "! parseLambda begin\n";
      std::string variable;
      Object *type;
      if (parseID(variable) || skipWhitespace() || parseChar(':') ||
          skipWhitespace() || parseTerm(type) || skipWhitespace() ||
          parseChar(']') || skipWhitespace())
        return failure();
      auto *atom = vm.create<Atom>(variable);

      // Add the binding.
      Bindings::Scope scope(bindings);
      bindings.insert(variable, atom);

      // Parse the body of the Lambda.
      Object *body;
      if (parseTerm(body))
        return failure();

      term = vm.create<Lambda>(atom, type, body);
      return success();
    }

    // Parse a single term.
    auto result = parseOptionalSubterm(term);
    if (!result.has_value() || *result)
      return result;

    std::cout << "! head is " << *term << "\n";

    // See if we have an application.
    while (true) {
      skipWhitespace();
      Object *argument;
      auto result = parseOptionalSubterm(argument);
      if (!result.has_value())
        break;
      if (*result)
        return failure();
      term = vm.create<Apply>(term, argument);
      std::cout << "! application: " << term;
    }

    if (succeeded(parseOptionalString("->"))) {
      std::cout << "! parse anon Pi begin\n";
      skipWhitespace();

      // Create an atom with no binding.
      auto variable = vm.create<Atom>("_");

      // Parse the body of the Pi.
      Object *body;
      if (parseTerm(body))
        return failure();

      std::cout << "! returned from parsing the body\n";
      std::cout << variable << "=" << *variable << "\n";
      std::cout << term << "=" << *term << "\n";
      std::cout << body << "=" << *body << "\n";
      term = vm.create<Pi>(variable, term, body);
      std::cout << term << "=??\n";
      std::cout << "! parse anon Pi end " << *term << "\n";

      return success();
    }

    return success();
  }

  ParseResult parseTerm(Object *&term) {
    std::cout << "! parseTerm begin\n";
    auto result = parseOptionalTerm(term);
    if (!result.has_value()) {
      std::cout << "expected term\n";
      return failure();
    }
    std::cout << "! parseTerm end\n";
    return *result;
  }

  /// sig  ::=                       % Empty signature
  ///        | decl sig              % Constant declaration
  ParseResult parseSig() {
    while (!code.empty()) {
      skipWhitespace();
      if (code.empty())
        return success();
      Object *decl;
      if (parseDecl(decl))
        return failure();
      std::cout << "parsed: " << *decl << "\n";
      decls.push_back(decl);
    }
    return success();
  }

  Module *parse() {
    Bindings::Scope scope(bindings);
    if (parseSig())
      return nullptr;
    return vm.create<Module>(std::move(decls));
  }

  VirtualMachine &vm;
  std::string_view code;
  Bindings bindings;
  std::vector<Object *> decls;
};
} // namespace

Module *aug::parse(VirtualMachine &vm, std::string_view code) {
  return Parser(vm, code).parse();
}
