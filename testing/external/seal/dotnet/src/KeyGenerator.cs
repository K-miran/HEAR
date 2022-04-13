﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

using Microsoft.Research.SEAL.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Microsoft.Research.SEAL
{
    /// <summary>
    /// Generates matching secret key and public key.
    /// </summary>
    /// <remarks>
    /// Generates matching secret key and public key. An existing KeyGenerator can
    /// also at any time be used to generate relinearization keys and Galois keys.
    /// Constructing a KeyGenerator requires only a SEALContext.
    /// </remarks>
    public class KeyGenerator : NativeObject
    {
        /// <summary>
        /// Creates a KeyGenerator initialized with the specified SEALContext.
        /// </summary>
        /// <remarks>
        /// Creates a KeyGenerator initialized with the specified <see cref="SEALContext" />.
        /// Dynamically allocated member variables are allocated from the global memory pool.
        /// </remarks>
        /// <param name="context">The SEALContext</param>
        /// <exception cref="ArgumentException">if encryption parameters are not
        /// valid</exception>
        /// <exception cref="ArgumentNullException">if context is null</exception>
        public KeyGenerator(SEALContext context)
        {
            if (null == context)
                throw new ArgumentNullException(nameof(context));
            if (!context.ParametersSet)
                throw new ArgumentException("Encryption parameters are not set correctly");

            NativeMethods.KeyGenerator_Create(context.NativePtr, out IntPtr ptr);
            NativePtr = ptr;
        }

        /// <summary>
        /// Creates an KeyGenerator instance initialized with the specified
        /// SEALContext and specified previously secret key.
        /// </summary>
        /// <remarks>
        /// Creates an KeyGenerator instance initialized with the specified
        /// SEALContext and specified previously secret key. This can e.g. be used
        /// to increase the number of relinearization keys from what had earlier
        /// been generated, or to generate Galois keys in case they had not been
        /// generated earlier.
        /// </remarks>
        /// <param name="context">The SEALContext</param>
        /// <param name="secretKey">A previously generated secret key</param>
        /// <exception cref="ArgumentNullException">if either context or secretKey
        /// are null</exception>
        /// <exception cref="ArgumentException">if encryption parameters are not
        /// valid</exception>
        /// <exception cref="ArgumentException">if secretKey or publicKey is not
        /// valid for encryption parameters</exception>
        public KeyGenerator(SEALContext context, SecretKey secretKey)
        {
            if (null == context)
                throw new ArgumentNullException(nameof(context));
            if (null == secretKey)
                throw new ArgumentNullException(nameof(secretKey));
            if (!context.ParametersSet)
                throw new ArgumentException("Encryption parameters are not set correctly");
            if (!ValCheck.IsValidFor(secretKey, context))
                throw new ArgumentException("Secret key is not valid for encryption parameters");

            NativeMethods.KeyGenerator_Create(context.NativePtr,
                secretKey.NativePtr, out IntPtr ptr);
            NativePtr = ptr;
        }

        /// <summary>
        /// Creates an KeyGenerator instance initialized with the specified
        /// SEALContext and specified previously secret and public keys.
        /// </summary>
        /// <remarks>
        /// Creates an KeyGenerator instance initialized with the specified
        /// SEALContext and specified previously secret and public keys. This can
        /// e.g. be used to increase the number of relinearization keys from what
        /// had earlier been generated, or to generate Galois keys in case they
        /// had not been generated earlier.
        /// </remarks>
        /// <param name="context">The SEALContext</param>
        /// <param name="secretKey">A previously generated secret key</param>
        /// <param name="publicKey">A previously generated public key</param>
        /// <exception cref="ArgumentNullException">if either context, secretKey
        /// or publicKey are null</exception>
        /// <exception cref="ArgumentException">if encryption parameters are not
        /// valid</exception>
        /// <exception cref="ArgumentException">if secretKey or publicKey is not
        /// valid for encryption parameters</exception>
        public KeyGenerator(SEALContext context, SecretKey secretKey, PublicKey publicKey)
        {
            if (null == context)
                throw new ArgumentNullException(nameof(context));
            if (null == secretKey)
                throw new ArgumentNullException(nameof(secretKey));
            if (null == publicKey)
                throw new ArgumentNullException(nameof(publicKey));
            if (!context.ParametersSet)
                throw new ArgumentException("Encryption parameters are not set correctly");
            if (!ValCheck.IsValidFor(secretKey, context))
                throw new ArgumentException("Secret key is not valid for encryption parameters");
            if (!ValCheck.IsValidFor(publicKey, context))
                throw new ArgumentException("Public key is not valid for encryption parameters");

            NativeMethods.KeyGenerator_Create(context.NativePtr, secretKey.NativePtr,
                publicKey.NativePtr, out IntPtr ptr);
            NativePtr = ptr;
        }

        /// <summary>
        /// Returns a copy of the public key.
        /// </summary>
        public PublicKey PublicKey
        {
            get
            {
                NativeMethods.KeyGenerator_PublicKey(NativePtr, out IntPtr pubKeyPtr);
                PublicKey pubKey = new PublicKey(pubKeyPtr);
                return pubKey;
            }
        }

        /// <summary>
        /// Returns a copy of the secret key.
        /// </summary>
        public SecretKey SecretKey
        {
            get
            {
                NativeMethods.KeyGenerator_SecretKey(NativePtr, out IntPtr secretKeyPtr);
                SecretKey secretKey = new SecretKey(secretKeyPtr);
                return secretKey;
            }
        }

        /// <summary>
        /// Generates and returns relinearization keys.
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// </summary>
        public RelinKeys RelinKeys()
        {
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            NativeMethods.KeyGenerator_RelinKeys(NativePtr, false, out IntPtr relinKeysPtr);
            return new RelinKeys(relinKeysPtr);
        }

        /// <summary>
        /// Generates and saves relinearization keys to an output stream.
        /// </summary>
        /// <remarks>
        /// Half of the polynomials in relinearization keys are randomly generated
        /// and are replaced with the seed used to compress output size. The output
        /// is in binary format and not human-readable. The output stream must have
        /// the "binary" flag set.
        /// </remarks>
        /// <param name="stream">The stream to save the relinearization keys to</param>
        /// <param name="comprMode">The desired compression mode</param>
        /// <exception cref="ArgumentNullException">if stream is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the stream is closed or does not
        /// support writing</exception>
        /// <exception cref="IOException">if I/O operations failed</exception>
        /// <exception cref="InvalidOperationException">if compression mode is not
        /// supported, or if compression failed</exception>

        public long RelinKeysSave(Stream stream, ComprModeType? comprMode = null)
        {
            if (null == stream)
                throw new ArgumentNullException(nameof(stream));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            comprMode = comprMode ?? Serialization.ComprModeDefault;
            if (!Serialization.IsSupportedComprMode(comprMode.Value))
                throw new InvalidOperationException("Unsupported compression mode");

            NativeMethods.KeyGenerator_RelinKeys(NativePtr, true, out IntPtr relinKeysPtr);
            using (RelinKeys relinKeys = new RelinKeys(relinKeysPtr))
            {
                return relinKeys.Save(stream, comprMode);
            }
        }

        /// <summary>
        /// Generates and returns Galois keys.
        /// </summary>
        /// <remarks>
        /// This function creates logarithmically many (in degree of the polynomial modulus)
        /// Galois keys that is sufficient to apply any Galois automorphism (e.g. rotations)
        /// on encrypted data. Most users will want to use this overload of the function.
        /// </remarks>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        public GaloisKeys GaloisKeys()
        {
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            NativeMethods.KeyGenerator_GaloisKeysAll(NativePtr, false, out IntPtr galoisKeysPtr);
            return new GaloisKeys(galoisKeysPtr);
        }

        /// <summary>
        /// Generates and saves Galois keys to an output stream.
        /// </summary>
        /// <remarks>
        /// This function creates logarithmically many (in degree of the polynomial modulus)
        /// Galois keys that is sufficient to apply any Galois automorphism (e.g. rotations)
        /// on encrypted data. Most users will want to use this overload of the function.
        ///
        /// Half of the polynomials in relinearization keys are randomly generated
        /// and are replaced with the seed used to compress output size. The output
        /// is in binary format and not human-readable. The output stream must have
        /// the "binary" flag set.
        /// </remarks>
        /// <param name="stream">The stream to save the Galois keys to</param>
        /// <param name="comprMode">The desired compression mode</param>
        /// <exception cref="ArgumentNullException">if stream is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the stream is closed or does not
        /// support writing</exception>
        /// <exception cref="IOException">if I/O operations failed</exception>
        /// <exception cref="InvalidOperationException">if compression mode is not
        /// supported, or if compression failed</exception>
        public long GaloisKeysSave(Stream stream, ComprModeType? comprMode = null)
        {
            if (null == stream)
                throw new ArgumentNullException(nameof(stream));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            comprMode = comprMode ?? Serialization.ComprModeDefault;
            if (!Serialization.IsSupportedComprMode(comprMode.Value))
                throw new InvalidOperationException("Unsupported compression mode");

            NativeMethods.KeyGenerator_GaloisKeysAll(NativePtr, true, out IntPtr galoisKeysPtr);
            using (GaloisKeys galoisKeys = new GaloisKeys(galoisKeysPtr))
            {
                return galoisKeys.Save(stream, comprMode);
            }
        }

        /// <summary>
        /// Generates and returns Galois keys.
        /// </summary>
        ///
        /// <remarks>
        /// This function creates specific Galois keys that can be used to apply specific
        /// Galois automorphisms on encrypted data. The user needs to give as input a vector
        /// of Galois elements corresponding to the keys that are to be created.
        ///
        /// The Galois elements are odd integers in the interval [1, M-1], where
        /// M = 2*N, and N = PolyModulusDegree. Used with batching, a Galois element
        /// 3^i % M corresponds to a cyclic row rotation i steps to the left, and
        /// a Galois element 3^(N/2-i) % M corresponds to a cyclic row rotation i
        /// steps to the right. The Galois element M-1 corresponds to a column rotation
        /// (row swap). In the polynomial view (not batching), a Galois automorphism by
        /// a Galois element p changes Enc(plain(x)) to Enc(plain(x^p)).
        /// </remarks>
        /// <param name="galoisElts">The Galois elements for which to generate keys</param>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the Galois elements are not valid</exception>
        public GaloisKeys GaloisKeys(IEnumerable<ulong> galoisElts)
        {
            if (null == galoisElts)
                throw new ArgumentNullException(nameof(galoisElts));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            ulong[] galoisEltsArr = galoisElts.ToArray();
            NativeMethods.KeyGenerator_GaloisKeysFromElts(NativePtr,
                (ulong)galoisEltsArr.Length, galoisEltsArr, false, out IntPtr galoisKeysPtr);
            return new GaloisKeys(galoisKeysPtr);
        }

        /// <summary>
        /// Generates and saves Galois keys to an output stream.
        /// </summary>
        ///
        /// <remarks>
        /// This function creates specific Galois keys that can be used to apply specific
        /// Galois automorphisms on encrypted data. The user needs to give as input a vector
        /// of Galois elements corresponding to the keys that are to be created.
        ///
        /// The Galois elements are odd integers in the interval [1, M-1], where
        /// M = 2*N, and N = PolyModulusDegree. Used with batching, a Galois element
        /// 3^i % M corresponds to a cyclic row rotation i steps to the left, and
        /// a Galois element 3^(N/2-i) % M corresponds to a cyclic row rotation i
        /// steps to the right. The Galois element M-1 corresponds to a column rotation
        /// (row swap). In the polynomial view (not batching), a Galois automorphism by
        /// a Galois element p changes Enc(plain(x)) to Enc(plain(x^p)).
        ///
        /// Half of the polynomials in relinearization keys are randomly generated
        /// and are replaced with the seed used to compress output size. The output
        /// is in binary format and not human-readable. The output stream must have
        /// the "binary" flag set.
        /// </remarks>
        /// <param name="galoisElts">The Galois elements for which to generate keys</param>
        /// <param name="stream">The stream to save the Galois keys to</param>
        /// <param name="comprMode">The desired compression mode</param>
        /// <exception cref="ArgumentNullException">if galoisElts or stream is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the Galois elements are not valid</exception>
        /// <exception cref="ArgumentException">if the stream is closed or does not
        /// support writing</exception>
        /// <exception cref="IOException">if I/O operations failed</exception>
        /// <exception cref="InvalidOperationException">if compression mode is not
        /// supported, or if compression failed</exception>
        public long GaloisKeysSave(IEnumerable<ulong> galoisElts, Stream stream, ComprModeType? comprMode = null)
        {
            if (null == stream)
                throw new ArgumentNullException(nameof(stream));
            if (null == galoisElts)
                throw new ArgumentNullException(nameof(galoisElts));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            comprMode = comprMode ?? Serialization.ComprModeDefault;
            if (!Serialization.IsSupportedComprMode(comprMode.Value))
                throw new InvalidOperationException("Unsupported compression mode");

            ulong[] galoisEltsArr = galoisElts.ToArray();
            NativeMethods.KeyGenerator_GaloisKeysFromElts(NativePtr,
                (ulong)galoisEltsArr.Length, galoisEltsArr, true, out IntPtr galoisKeysPtr);

            using (GaloisKeys galoisKeys = new GaloisKeys(galoisKeysPtr))
            {
                return galoisKeys.Save(stream, comprMode);
            }
        }

        /// <summary>
        /// Generates and returns Galois keys.
        /// </summary>
        /// <remarks>
        /// This function creates specific Galois keys that can be used to apply
        /// specific Galois automorphisms on encrypted data. The user needs to give
        /// as input a vector of desired Galois rotation step counts, where negative
        /// step counts correspond to rotations to the right and positive step counts
        /// correspond to rotations to the left. A step count of zero can be used to
        /// indicate a column rotation in the BFV scheme complex conjugation in the
        /// CKKS scheme.
        /// </remarks>
        /// <param name="steps">The rotation step counts for which to generate keys</param>
        /// <exception cref="ArgumentNullException">if steps is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the step counts are not valid</exception>
        public GaloisKeys GaloisKeys(IEnumerable<int> steps)
        {
            if (null == steps)
                throw new ArgumentNullException(nameof(steps));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            int[] stepsArr = steps.ToArray();
            NativeMethods.KeyGenerator_GaloisKeysFromSteps(NativePtr,
                (ulong)stepsArr.Length, stepsArr, false, out IntPtr galoisKeysPtr);
            return new GaloisKeys(galoisKeysPtr);
        }

        /// <summary>
        /// Generates and saves Galois keys to an output stream.
        /// </summary>
        /// <remarks>
        /// This function creates specific Galois keys that can be used to apply
        /// specific Galois automorphisms on encrypted data. The user needs to give
        /// as input a vector of desired Galois rotation step counts, where negative
        /// step counts correspond to rotations to the right and positive step counts
        /// correspond to rotations to the left. A step count of zero can be used to
        /// indicate a column rotation in the BFV scheme complex conjugation in the
        /// CKKS scheme.
        ///
        /// Half of the polynomials in relinearization keys are randomly generated
        /// and are replaced with the seed used to compress output size. The output
        /// is in binary format and not human-readable. The output stream must have
        /// the "binary" flag set.
        /// </remarks>
        /// <param name="steps">The rotation step counts for which to generate keys</param>
        /// <param name="stream">The stream to save the Galois keys to</param>
        /// <param name="comprMode">The desired compression mode</param>
        /// <exception cref="ArgumentNullException">if steps or stream is null</exception>
        /// <exception cref="InvalidOperationException">if the encryption parameters
        /// do not support batching and scheme is SchemeType.BFV</exception>
        /// <exception cref="InvalidOperationException">if the encryption
        /// parameters do not support keyswitching</exception>
        /// <exception cref="ArgumentException">if the step counts are not valid</exception>
        /// <exception cref="ArgumentException">if the stream is closed or does not
        /// support writing</exception>
        /// <exception cref="IOException">if I/O operations failed</exception>
        /// <exception cref="InvalidOperationException">if compression mode is not
        /// supported, or if compression failed</exception>
        public long GaloisKeysSave(IEnumerable<int> steps, Stream stream, ComprModeType? comprMode = null)
        {
            if (null == stream)
                throw new ArgumentNullException(nameof(stream));
            if (null == steps)
                throw new ArgumentNullException(nameof(steps));
            if (!UsingKeyswitching())
                throw new InvalidOperationException("Encryption parameters do not support keyswitching");

            comprMode = comprMode ?? Serialization.ComprModeDefault;
            if (!Serialization.IsSupportedComprMode(comprMode.Value))
                throw new InvalidOperationException("Unsupported compression mode");

            int[] stepsArr = steps.ToArray();
            NativeMethods.KeyGenerator_GaloisKeysFromSteps(NativePtr,
                (ulong)stepsArr.Length, stepsArr, true, out IntPtr galoisKeysPtr);

            using (GaloisKeys galoisKeys = new GaloisKeys(galoisKeysPtr))
            {
                return galoisKeys.Save(stream, comprMode);
            }
        }

        /// <summary>
        /// Destroy native object.
        /// </summary>
        protected override void DestroyNativeObject()
        {
            NativeMethods.KeyGenerator_Destroy(NativePtr);
        }

        internal bool UsingKeyswitching()
        {
            NativeMethods.KeyGenerator_ContextUsingKeyswitching(NativePtr, out bool result);
            return result;
        }
    }
}
