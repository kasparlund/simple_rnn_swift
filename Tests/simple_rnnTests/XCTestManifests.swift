import XCTest

#if !os(macOS)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(simple_rnnTests.allTests),
    ]
}
#endif
