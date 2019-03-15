import Foundation
let data_url = URL(fileURLWithPath: "/Users/kasparlund/AICodeData/swift/simple_rnn_swift/data/sherlock_holmes_the_canon.txt")
let data       = try! Array<Character>(String(NSString(contentsOfFile: data_url.path, encoding: String.Encoding.ascii.rawValue)))
//rnn_swift(data:data)
rnn_python(data:data)
