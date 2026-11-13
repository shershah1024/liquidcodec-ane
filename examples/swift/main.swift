// usage: decode MODELS_DIR CODES.bin T OUT.wav     (CODES.bin: int32 little-endian [12, T], row-major, codebook-major)
// build: swiftc -O -o decode examples/swift/LiquidANEDecoder.swift examples/swift/main.swift
import Foundation
let args = CommandLine.arguments
guard args.count == 5, let T = Int(args[3]) else { print("usage: decode MODELS_DIR CODES.bin T OUT.wav"); exit(2) }
let d = try Data(contentsOf: URL(fileURLWithPath: args[2])); let codes = d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
precondition(codes.count == 12 * T, "expected 12*T codes")
let t0 = Date(); let dec = try LiquidANEDecoder(dir: URL(fileURLWithPath: args[1])); let tLoad = Date().timeIntervalSince(t0)
let t1 = Date(); let y = try dec.decode(codes: codes, T: T); let tDec = Date().timeIntervalSince(t1)
// 16-bit PCM wav, 24 kHz mono
var w = Data(); func u32(_ v: UInt32) { var x = v.littleEndian; w.append(Data(bytes: &x, count: 4)) }; func u16(_ v: UInt16) { var x = v.littleEndian; w.append(Data(bytes: &x, count: 2)) }
w.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + 2 * y.count)); w.append("WAVEfmt ".data(using: .ascii)!); u32(16); u16(1); u16(1); u32(24000); u32(48000); u16(2); u16(16)
w.append("data".data(using: .ascii)!); u32(UInt32(2 * y.count)); for s in y { u16(UInt16(bitPattern: Int16(max(-1, min(1, s)) * 32767))) }
try w.write(to: URL(fileURLWithPath: args[4]))
print(String(format: "wrote %@: %.2f s audio; load %.2f s; decode %.3f s (%.0fx real time)", args[4], Double(y.count) / 24000, tLoad, tDec, Double(y.count) / 24000 / tDec))
